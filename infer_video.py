#!/usr/bin/env python3
"""Run NeighborNet scene detection on a standalone video.

Pipeline:
  1. Shot boundary detection  (PySceneDetect ContentDetector)
  2. Feature extraction        (ResNet-50 pretrained on ImageNet -> 2048-d)
  3. Neighbor graph construction (cosine-similarity top-k within windows)
  4. NeighborNet sliding-window inference
  5. Scene boundary output with timestamps

Usage:
  python infer_video.py --video path/to/video.mp4 --checkpoint epoch_10.pth.tar
  python infer_video.py --video path/to/video.mp4 --checkpoint epoch_10.pth.tar --device cpu
  python infer_video.py --video path/to/video.mp4 --checkpoint epoch_10.pth.tar --threshold 0.4
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.NeighborNet import SLNet
from dataloader.BaseDataset import BaseDataset


# ---------------------------------------------------------------------------
# 1. Shot boundary detection
# ---------------------------------------------------------------------------

def detect_shots(video_path, threshold=27.0):
    """Return a list of shots as dicts with start/end frame and time."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=True)
    scene_list = sm.get_scene_list()

    if not scene_list:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [{
            'start_frame': 0,
            'end_frame': max(total - 1, 0),
            'start_time': 0.0,
            'end_time': max(total - 1, 0) / fps,
        }]

    shots = []
    for start_tc, end_tc in scene_list:
        shots.append({
            'start_frame': start_tc.get_frames(),
            'end_frame': end_tc.get_frames() - 1,
            'start_time': start_tc.get_seconds(),
            'end_time': end_tc.get_seconds(),
        })
    return shots


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

def extract_middle_frames(video_path, shots):
    """Extract the middle frame of every shot."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for shot in tqdm(shots, desc='Extracting frames'):
        mid = (shot['start_frame'] + shot['end_frame']) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


def extract_features(frames, device, batch_size=32):
    """Run ResNet-50 (minus FC) on frames -> (n_shots, 2048)."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size), desc='Extracting features'):
            batch_imgs = frames[i:i + batch_size]
            batch_t = torch.stack([tfm(Image.fromarray(f)) for f in batch_imgs])
            batch_t = batch_t.to(device)
            feats = resnet(batch_t).squeeze(-1).squeeze(-1)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# 3. Neighbor graph construction
# ---------------------------------------------------------------------------

def compute_neighbor_links(features, seg_sz=20, topk=5):
    """For each shot, build a seg_sz-wide window and find top-k cosine
    neighbors per position within that window."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feat_norm = features / norms
    n_shots = len(features)
    half = seg_sz // 2
    all_links = []

    for center in range(n_shots):
        ctx_ids = np.arange(center - half + 1, center + half + 1)
        ctx_ids = np.clip(ctx_ids, 0, n_shots - 1)
        win = feat_norm[ctx_ids]
        sim = win @ win.T

        links = []
        for i in range(seg_sz):
            s = sim[i].copy()
            s[i] = -np.inf
            top_idx = np.argsort(s)[-topk:][::-1].tolist()
            links.append(top_idx)
        all_links.append(links)

    return all_links


def build_graph_tensors(links, topk=5):
    """Convert neighbor-link list into adjacency and index tensors."""
    helper = BaseDataset([])
    gh = helper._build_graph(links)
    inx = helper._index_matric(links, n_top=topk)
    rlink = helper._gen_reason_link(links)
    rgh = helper._build_graph(rlink)

    hop = torch.from_numpy(np.stack([gh, rgh], axis=0)).float()
    inxs = torch.from_numpy(inx[None]).float()
    return hop, inxs


# ---------------------------------------------------------------------------
# 4. Model loading (auto-detect architecture from checkpoint keys)
# ---------------------------------------------------------------------------

def detect_model_config(state_dict):
    """Infer architecture hyper-parameters from checkpoint keys."""
    embed_dim = state_dict['proj.linear_1.weight'].shape[0]
    pos_features = state_dict['embed_pos.w'].shape[1] + 1
    has_mha = any('att_nei.mha' in k for k in state_dict)
    variant = 'm' if has_mha else 'orig'
    has_fuse = any('fuse' in k for k in state_dict)
    is_pretrain = any('detect.detect' in k for k in state_dict)
    mode = 'pretrain' if is_pretrain else 'fine'

    topk = 5
    if 's1.att_nei.nn.weight_v' in state_dict:
        topk = state_dict['s1.att_nei.nn.weight_v'].shape[1]

    return dict(
        embed_dim=embed_dim,
        pos_features=pos_features,
        variant=variant,
        has_fuse=has_fuse,
        mode=mode,
        topk=topk,
    )


def load_model(checkpoint_path, device, seg_sz=20, tnei=2):
    """Load checkpoint, build a matching SLNet, return (model, config)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    cfg = detect_model_config(sd)

    print(f'  embed_dim={cfg["embed_dim"]}, pos_features={cfg["pos_features"]}, '
          f'variant={cfg["variant"]}, mode={cfg["mode"]}, topk={cfg["topk"]}, '
          f'has_fuse={cfg["has_fuse"]}')
    if cfg['mode'] == 'pretrain':
        print('  Checkpoint type: self-supervised (BaSSLDet head)')
    else:
        print('  Checkpoint type: supervised (LatentDetector head)')

    model = SLNet(
        shot_dim=2048,
        embed_dim=cfg['embed_dim'],
        pos_features=cfg['pos_features'],
        att_drop=0.1,
        topk=cfg['topk'],
        seg_sz=seg_sz,
        tnei=tnei,
        mode=cfg['mode'],
        variant=cfg['variant'],
        has_fuse=cfg['has_fuse'],
    )
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, cfg


# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------

def run_inference(model, features, all_links, cfg, device, seg_sz=20):
    """Slide a window over all shots and collect per-shot boundary scores."""
    topk = cfg['topk']
    n_shots = len(features)
    half = seg_sz // 2
    center_idx = half - 1
    is_pretrain = cfg['mode'] == 'pretrain'

    predictions = np.zeros(n_shots)

    with torch.no_grad():
        for c in tqdm(range(n_shots), desc='Running NeighborNet'):
            ctx_ids = np.arange(c - half + 1, c + half + 1)
            ctx_ids = np.clip(ctx_ids, 0, n_shots - 1)
            win_feats = features[ctx_ids]

            x = torch.from_numpy(win_feats).float().unsqueeze(0).to(device)
            hop, inxs = build_graph_tensors(all_links[c], topk=topk)
            hop = hop.unsqueeze(0).to(device)
            inxs = inxs.unsqueeze(0).to(device)

            pred = model(x, hop, inxs)

            if is_pretrain:
                p = pred.cpu().numpy()
                predictions[c] = float(p[center_idx]) if p.ndim >= 1 else float(p)
            else:
                predictions[c] = float(pred.cpu().numpy())

    return predictions


# ---------------------------------------------------------------------------
# 6. Post-processing and output
# ---------------------------------------------------------------------------

def predictions_to_scenes(predictions, shots, threshold=0.5):
    """Group shots into scenes based on boundary scores."""
    n = len(predictions)
    scenes = []
    start = 0
    for i in range(n):
        if predictions[i] > threshold or i == n - 1:
            scenes.append({
                'scene_id': len(scenes),
                'start_shot': start,
                'end_shot': i,
                'start_time': shots[start]['start_time'],
                'end_time': shots[i]['end_time'],
                'n_shots': i - start + 1,
            })
            start = i + 1
    return scenes


def save_shot_frames(frames, shots, outdir):
    """Save shot keyframes as images."""
    shots_dir = os.path.join(outdir, 'shots')
    os.makedirs(shots_dir, exist_ok=True)
    for i, (frame, shot) in enumerate(zip(frames, shots)):
        fname = os.path.join(shots_dir, f'shot_{i:04d}_'
                             f'{shot["start_time"]:.2f}s-{shot["end_time"]:.2f}s.jpg')
        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f'  Saved {len(frames)} shot keyframes to {shots_dir}/')


def save_scene_clips(video_path, scenes, outdir, reencode=False, workers=4):
    """Cut scene clips from the source video using ffmpeg.

    By default uses stream-copy (``-c copy``) which is near-instant but snaps
    to the nearest keyframe.  Pass ``reencode=True`` for frame-accurate cuts
    (much slower).
    """
    import subprocess
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not shutil.which('ffmpeg'):
        print('  ffmpeg not found -- skipping scene clip export')
        return

    scenes_dir = os.path.join(outdir, 'scenes')
    os.makedirs(scenes_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    frame_dur = 1.0 / fps

    def _cut(sc):
        start = sc['start_time']
        end = sc['end_time'] - frame_dur
        duration = max(end - start, frame_dur)
        out_path = os.path.join(
            scenes_dir,
            f'scene_{sc["scene_id"]:03d}_{start:.2f}s-{sc["end_time"]:.2f}s.mp4')
        if reencode:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start), '-i', video_path,
                '-t', f'{duration:.6f}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', '1',
                out_path,
            ]
        else:
            # -ss AFTER -i: starts from the first keyframe >= start_time,
            # avoiding the extra-shot problem of pre-input seeking.
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start),
                '-t', f'{duration:.6f}',
                '-c', 'copy',
                '-avoid_negative_ts', '1',
                out_path,
            ]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_cut, sc) for sc in scenes]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Cutting scene clips'):
            f.result()

    print(f'  Saved {len(scenes)} scene clips to {scenes_dir}/')


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02d}:{int(m):02d}:{s:05.2f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='NeighborNet standalone video scene detection')
    ap.add_argument('--video', required=True, help='Path to input video')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to .pth.tar checkpoint')
    ap.add_argument('--device', default='cuda:0',
                    help='torch device (default: cuda:0)')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Boundary score threshold (default: 0.5)')
    ap.add_argument('--shot_threshold', type=float, default=27.0,
                    help='PySceneDetect ContentDetector threshold (default: 27.0)')
    ap.add_argument('--seg_sz', type=int, default=20,
                    help='Sliding window size in shots (default: 20)')
    ap.add_argument('--outdir', default=None,
                    help='Output directory (default: output/<video_name>/)')
    ap.add_argument('--save_clips', action='store_true',
                    help='Also cut scene clips via ffmpeg')
    args = ap.parse_args()

    if args.device != 'cpu' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'Using device: {device}')

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    outdir = args.outdir or os.path.join('output', video_stem)
    os.makedirs(outdir, exist_ok=True)
    print(f'Output directory: {outdir}/')

    # --- Step 1: Shot detection ---
    print('\n[1/6] Detecting shots ...')
    shots = detect_shots(args.video, threshold=args.shot_threshold)
    n_shots = len(shots)
    print(f'  Found {n_shots} shots')
    if n_shots < args.seg_sz:
        print(f'  Warning: only {n_shots} shots (< seg_sz={args.seg_sz}). '
              'Edge padding will be applied.')

    # --- Step 2: Feature extraction ---
    print('\n[2/6] Extracting ResNet-50 features ...')
    frames = extract_middle_frames(args.video, shots)
    features = extract_features(frames, device)
    print(f'  Features shape: {features.shape}')

    # --- Step 3: Save shot keyframes ---
    print('\n[3/6] Saving shot keyframes ...')
    save_shot_frames(frames, shots, outdir)

    # --- Step 4: Load model ---
    print('\n[4/6] Loading NeighborNet model ...')
    model, cfg = load_model(args.checkpoint, device, seg_sz=args.seg_sz)

    # --- Step 5: Build graphs and run inference ---
    print('\n[5/6] Building neighbor graphs and running inference ...')
    all_links = compute_neighbor_links(
        features, seg_sz=args.seg_sz, topk=cfg['topk'])
    predictions = run_inference(
        model, features, all_links, cfg, device, seg_sz=args.seg_sz)

    # --- Step 6: Results ---
    print('\n[6/6] Post-processing and saving results ...')
    scenes = predictions_to_scenes(predictions, shots, threshold=args.threshold)

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'Detected {len(scenes)} scenes  (threshold={args.threshold})')
    print(sep)
    for sc in scenes:
        print(f'  Scene {sc["scene_id"]:3d}:  '
              f'{format_time(sc["start_time"])} -> {format_time(sc["end_time"])}  '
              f'({sc["n_shots"]} shots)')

    result = {
        'video': os.path.abspath(args.video),
        'checkpoint': os.path.abspath(args.checkpoint),
        'threshold': args.threshold,
        'n_shots': n_shots,
        'n_scenes': len(scenes),
        'shots': shots,
        'scenes': scenes,
        'per_shot_scores': predictions.tolist(),
    }
    results_path = os.path.join(outdir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Results JSON saved to {results_path}')

    if args.save_clips:
        print('\n  Cutting scene clips ...')
        save_scene_clips(args.video, scenes, outdir)

    print(f'\nAll outputs saved to {outdir}/')


if __name__ == '__main__':
    main()
