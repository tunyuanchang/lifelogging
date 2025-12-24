import os
import sys
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import subprocess

# ==============================
# CONFIG
# ==============================
OUTPUT_DIR = "/home/tunyuan/divexplore/backend/output"
SCENE_THRESHOLD = 30.0
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"  # you can switch to Qwen3-VL when available

# ==============================
# LOAD MODEL
# ==============================
print("Loading Qwen3-VL model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto"
)

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

def main(VIDEO_PATH):
    BASENAME = os.path.basename(VIDEO_PATH).split('.')[0]

    SCENE_DIR = os.path.join(OUTPUT_DIR, "scene")
    CAPTION_DIR = os.path.join(OUTPUT_DIR, "caption")
    TIMESTAMP_DIR = os.path.join(OUTPUT_DIR, "timestamp")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCENE_DIR, exist_ok=True)
    os.makedirs(CAPTION_DIR, exist_ok=True)
    os.makedirs(TIMESTAMP_DIR, exist_ok=True)

    # ==============================
    # STEP 1: Scene Detection and Split Scenes
    # ==============================
    print("Detecting scenes...")
    scene_list = detect(VIDEO_PATH, ContentDetector(threshold=SCENE_THRESHOLD))
    print(f"Detected {len(scene_list)} scenes.")

    split_video_ffmpeg(VIDEO_PATH, scene_list, output_dir=SCENE_DIR)
    
    # ==============================
    # STEP 2: Summarize Each Scene
    # ==============================

    print("Generating caption...")
    scene_summaries = []

    for idx, scene in enumerate(scene_list):
        start, end = scene
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()

        start_frame = start.get_frames()
        end_frame = end.get_frames()

        # print(f"\nScene {idx+1}: {start_frame} → {end_frame}")

        scene_path = os.path.join(SCENE_DIR, f"{BASENAME}-Scene-{idx+1:03d}.mp4")
        # print(scene_path)

        # ==========================
        # Create messages
        # ==========================
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{scene_path}",
                        "fps": 2.0,
                        "resized_height": 280,
                        "resized_width": 280
                    },
                    {
                        "type": "text",
                        "text": f"Summarize this video scene ({start_sec:.1f}s-{end_sec:.1f}s) in 2-3 sentences."
                    }
                ],
            }
        ]

        # Process vision info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(model.device)

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        prompt_len = inputs["input_ids"].shape[-1]
        assistant_ids = generated_ids[0][prompt_len:]

        result = processor.decode(
            assistant_ids,
            skip_special_tokens=True
        )
        # result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(f"Scene {idx+1} summary:\n{result}\n")

        scene_summaries.append({
            "scene": idx + 1,
            "start": start_frame,
            "end": end_frame,
            "summary": result
        })

    # ==============================
    # STEP 3: Combine Overall Summary
    # ==============================
    scene_text = "\n".join(
        [f"Scene {s['scene']} ({s['start']} - {s['end']}): {s['summary']}" for s in scene_summaries]
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here are the summaries of each scene:\n{scene_text}\n\nPlease write:\n1. A short TL;DR (2 sentences)\n2. A full summary (5-7 sentences)."}
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    prompt_len = inputs["input_ids"].shape[-1]
    assistant_ids = generated_ids[0][prompt_len:]

    final_result = processor.decode(
        assistant_ids,
        skip_special_tokens=True
    )
    # final_result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # print("\nFinal video summary:\n", final_result)

    # ==============================
    # STEP 4: Save Results
    # ==============================
    output_path = os.path.join(TIMESTAMP_DIR, f"{BASENAME}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in scene_summaries:
            f.write(f"{s['start']} {s['end']}\n")

    output_path = os.path.join(CAPTION_DIR, f"{BASENAME}_summary.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Scene Summaries ===")
        for s in scene_summaries:
            f.write(f"\nScene-{s['scene']:03d}\n{s['start']}-{s['end']}\n{s['summary']}\n")
        f.write("\n=== Final Summary ===\n")
        f.write(final_result)

    print(f"\n(Scene) {SCENE_DIR}\n(Caption) {output_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <VIDEO_PATH>")
        sys.exit(1)
    
    VIDEO_PATH = sys.argv[1]
    main(VIDEO_PATH)