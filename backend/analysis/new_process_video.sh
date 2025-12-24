#!/bin/bash

# Check input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <full_path_to_video>"
    exit 1
fi

# Input full video path
VIDEO_PATH="$1"
VIDEO_FILENAME=$(basename "$VIDEO_PATH")
BASENAME="${VIDEO_FILENAME%.*}"

# Output paths
SCENES_DIR="../output/scenes"
KEYFRAMES_DIR="../output/keyframes/$BASENAME"
SCENE_FILE="$SCENES_DIR/${VIDEO_FILENAME}.scenes.txt"

# 0. Get Video FPS
echo "=== Video FPS ==="
python3 get_fps.py "$VIDEO_PATH"

# 1. Shot detection & Caption generation
echo "=== Shot Detection & Caption Generation ==="
python3 new_caption.py "$VIDEO_PATH"

# 2. ASR extraction
echo "=== ASR Extraction ==="
python3 extract_asr.py "$VIDEO_PATH"

# 3. SI calculation
echo "=== SI Calculation ==="
python3 calculate_si.py "$VIDEO_PATH"

echo "=== ✅ Finished processing: $VIDEO_FILENAME ==="

