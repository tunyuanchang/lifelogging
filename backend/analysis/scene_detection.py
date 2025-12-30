import os
import sys
import math
import torch
import open_clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
frame_skip = 1 
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
save_path = "result.png"
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot(cos_sims, segments, frame_indices, boundary_val):
    """
    Plot cosine similarity with segments alternating in color.
    Even segments = Royal Blue, Odd segments = Firebrick Red.
    """
    plt.figure(figsize=(15, 6))
    
    # Initialize color array with object type for Matplotlib 3.9 compatibility
    colors = np.array([''] * len(cos_sims), dtype='object')
    
    # We map segments to the cos_sims array
    # cos_sims[i] represents the transition between frame_indices[i] and frame_indices[i+1]
    for i, (start_f, end_f) in enumerate(segments):
        # Determine color based on segment index
        seg_color = 'royalblue' if i % 2 == 0 else 'firebrick'
        
        # Find which indices in the cos_sims array fall within this segment's range
        # A segment from frame A to B covers transitions at indices where frame_indices is between A and B
        mask = (np.array(frame_indices[:-1]) >= start_f) & (np.array(frame_indices[:-1]) < end_f)
        colors[mask] = seg_color

    # Handle any unassigned bars (safety fallback)
    colors[colors == ''] = 'gray'

    x = np.arange(len(cos_sims))
    # Pass as list to avoid NumPy string-width issues in Matplotlib 3.9
    plt.bar(x, cos_sims, color=colors.tolist(), width=1.0, alpha=0.9)
    
    # Visual aids
    # plt.axhline(y=boundary_val, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({boundary_val:.4f})')
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Frame Index")
    plt.title(f"Scene Detection ({len(segments)} scenes in total)")
    # plt.legend(loc='lower left')
    
    # Set Y-axis to focus on the data range
    # if len(cos_sims) > 0:
    #     plt.ylim(max(0, np.min(cos_sims) - 0.02), 1.01)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def main():
    # --- 1. Get arguments with Defaults ---
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path> [similarity_percentile] [min_threshold_sec]")
        sys.exit(1)

    video_path = sys.argv[1]
    # Default: 0.95 (we look for the bottom 5% of similarities)
    sim_percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    # Default: 2.0 seconds minimum segment duration
    min_duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    # --- 2. Load OpenCLIP ---
    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device).eval()

    # --- 3. Process Video ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not open video or FPS is 0.")
        return
    
    min_frames = int(min_duration * fps)
    cos_sims = []
    frame_indices = []
    prev_emb = None
    frame_idx = 0

    print(f"Analyzing video: {os.path.basename(video_path)} ({fps} FPS)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_indices.append(frame_idx)

            # Feature extraction
            img = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)

            if prev_emb is not None:
                sim = torch.nn.functional.cosine_similarity(prev_emb, emb).item()
                cos_sims.append(sim)
            prev_emb = emb

        frame_idx += 1
    cap.release()

    cos_sims = np.array(cos_sims)
    if len(cos_sims) == 0:
        print("No frames processed.")
        return

    # --- 4. Detect Potential Changes (Bottom X%) ---
    n = cos_sims.size
    k = math.ceil(n * (1 - sim_percentile))
    k = max(0, min(k, n - 1))
    
    boundary = np.partition(cos_sims, k)[k]
    # Potential cut indices (where similarity is below the boundary)
    raw_cuts = np.where(cos_sims <= boundary)[0] + 1

    # --- 5. Generate Segments with Min-Duration Filter ---
    segments = []
    start_ptr = 0 # This is an index in frame_indices
    
    for cut_idx in raw_cuts:
        # cut_idx corresponds to frame_indices[cut_idx]
        # Check if the segment is long enough
        num_frames = frame_indices[cut_idx] - frame_indices[start_ptr]
        
        if num_frames >= min_frames:
            segments.append((frame_indices[start_ptr], frame_indices[cut_idx]))
            start_ptr = cut_idx

    # Add the final segment
    segments.append((frame_indices[start_ptr], frame_indices[-1]))

    print(f"\nFinal Result:")
    print(f"Total frame pairs: {n}")
    print(f"Similarity boundary: {boundary:.4f}")
    print(f"Potential segements: {np.sum(cos_sims <= boundary)}")
    print(f"Segments detected: {len(segments)}")
    
    # for i, seg in enumerate(segments):
    #     print(f"  Segment {i}: Frames {seg[0]} to {seg[1]}")

    # --- 6. Plotting ---
    plot(cos_sims, segments, frame_indices, boundary)

if __name__ == "__main__":
    main()