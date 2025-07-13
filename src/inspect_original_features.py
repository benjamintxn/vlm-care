#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Configuration
ORIG_FEAT_DIR = './data/features'
SPLITS = ['training', 'testing']

def analyze_features(features, detections):
    """Perform detailed analysis of feature and detection data"""
    stats = {
        'feat_shape': features.shape,
        'det_shape': detections.shape,
        'feat_min': features.min(),
        'feat_max': features.max(),
        'feat_mean': features.mean(),
        'feat_std': features.std(),
        'det_min': detections[..., :4].min(),  # Only bbox coords
        'det_max': detections[..., :4].max(),
        'det_mean': detections[..., :4].mean(),
        'non_zero_objects': np.count_nonzero(np.any(features, axis=(0, 2))) / features.shape[1]
    }
    return stats

def plot_feature_distribution(features, video_id):
    """Visualize feature distribution for a video clip"""
    plt.figure(figsize=(12, 6))
    
    # Global feature distribution (object 0)
    plt.subplot(1, 2, 1)
    global_feats = features[:, 0, :].flatten()
    plt.hist(global_feats, bins=50, alpha=0.7, color='blue')
    plt.title(f'Global Features ({video_id})')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Local object feature distribution
    plt.subplot(1, 2, 2)
    local_feats = features[:, 1:, :].flatten()
    plt.hist(local_feats, bins=50, alpha=0.7, color='green')
    plt.title(f'Local Object Features ({video_id})')
    plt.xlabel('Feature Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./outputs/feature_analysis/{video_id}_feature_dist.png')
    plt.close()

def extract_base_video_id(clip_id):
    """Extract base video ID from clip ID"""
    # Handle formats like '000003' or '000003_1'
    match = re.match(r'(\d{6})(?:_\d+)?', clip_id)
    return match.group(1) if match else clip_id

def main():
    # Create output directory
    os.makedirs('./outputs/feature_analysis', exist_ok=True)
    
    # Initialize statistics collectors
    clip_stats = []
    base_video_stats = defaultdict(lambda: {'clip_count': 0, 'feat_means': []})
    object_counts = defaultdict(int)
    
    for split in SPLITS:
        split_dir = os.path.join(ORIG_FEAT_DIR, split)
        batch_files = sorted(glob.glob(os.path.join(split_dir, 'batch_*.npz')))
        
        if not batch_files:
            print(f"No batch_*.npz files found in {split_dir}")
            continue
        
        print(f"\nAnalyzing {split} split ({len(batch_files)} batch files)")
        
        for batch_file in batch_files:
            try:
                with np.load(batch_file, allow_pickle=True) as data:
                    feats = data['data']    # (B, n_frames, n_det, feat_dim)
                    dets = data['det']      # (B, n_frames, n_det, 6)
                    clip_ids = data['ID']   # array of clip IDs
                    
                    # Convert IDs to strings
                    str_ids = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in clip_ids]
                    
                    # Record object count
                    object_counts[feats.shape[2]] += feats.shape[0]
                    
                    for i, clip_id in enumerate(str_ids):
                        base_id = extract_base_video_id(clip_id)
                        
                        # Basic stats for this clip
                        stats = analyze_features(feats[i], dets[i])
                        stats['clip_id'] = clip_id
                        stats['base_video_id'] = base_id
                        stats['split'] = split
                        stats['batch_file'] = os.path.basename(batch_file)
                        clip_stats.append(stats)
                        
                        # Aggregate stats for base video
                        base_video_stats[base_id]['clip_count'] += 1
                        base_video_stats[base_id]['feat_means'].append(stats['feat_mean'])
                        
                        # Print detailed info for first few clips
                        if base_video_stats[base_id]['clip_count'] <= 2:
                            print(f"\nFound clip {clip_id} (base {base_id}) in {batch_file} (index {i}):")
                            print(f"  Feature shape: {feats[i].shape}")
                            print(f"  Detection shape: {dets[i].shape}")
                            print(f"  Features - min: {stats['feat_min']:.4f}, max: {stats['feat_max']:.4f}, mean: {stats['feat_mean']:.4f}")
                            print(f"  BBox coords - min: {stats['det_min']:.1f}, max: {stats['det_max']:.1f}, mean: {stats['det_mean']:.1f}")
                            print(f"  Non-zero objects: {stats['non_zero_objects']*100:.1f}%")
                            
                            # Plot feature distribution
                            plot_feature_distribution(feats[i], clip_id)
            
            except Exception as e:
                print(f"Error processing {batch_file}: {str(e)}")
    
    # Print overall statistics
    print("\n===== Dataset Summary =====")
    print(f"Total clips analyzed: {len(clip_stats)}")
    print(f"Unique base videos: {len(base_video_stats)}")
    print(f"Object dimensions found: {list(object_counts.keys())}")
    
    # Calculate clips per video
    clips_per_video = [stats['clip_count'] for stats in base_video_stats.values()]
    
    # Print distribution of clips per video
    print("\n===== Clips per Video =====")
    print(f"Min clips: {min(clips_per_video)}")
    print(f"Max clips: {max(clips_per_video)}")
    print(f"Avg clips: {sum(clips_per_video)/len(clips_per_video):.1f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(clips_per_video, bins=20, alpha=0.7, color='purple')
    plt.title('Clips per Base Video')
    plt.xlabel('Number of Clips')
    plt.ylabel('Number of Videos')
    plt.grid(axis='y')
    plt.savefig('./outputs/feature_analysis/clips_per_video_dist.png')
    plt.close()

if __name__ == '__main__':
    main()