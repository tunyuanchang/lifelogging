import os
import sys
import cv2
import numpy as np
import argparse
import csv

MAX_RESOLUTION = 25.0
MAX_QUALITY = 1.0
OUTPUT_DIR = "/home/tunyuan/divexplore/backend/output/si"

def parse_scenes_file(scenes_file):
    scenes = []
    with open(scenes_file, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split())
            scenes.append((start, end))
    return scenes
    
def main(VIDEO_PATH, SCENE_PATH):
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    si_values = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        si_frame = np.std(sobel_magnitude)
        
        si_values.append(si_frame)
        frame_count += 1
        
        # if frame_count % 2000 == 0:
        #     print(f"Processed {frame_count} frames...")
    
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    BASENAME = os.path.basename(VIDEO_PATH).split('.')[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_folder = os.path.join(OUTPUT_DIR, BASENAME)
    os.makedirs(output_folder, exist_ok=True)

    scenes_file = os.path.join(SCENE_PATH, f"{BASENAME}.txt")
    scene_list = parse_scenes_file(scenes_file)

    if si_values:
        data_to_write = [(f"{BASENAME}-{i+1}", MAX_RESOLUTION, MAX_QUALITY, si) for i, si in enumerate(si_values)]

        output_path = os.path.join(output_folder, f"image_si.csv")

        with open(output_path, 'w', newline='') as csvfile:
            # Define the writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Frame_Number', 'Resolution', 'Quality', 'SI_Value'])
            csv_writer.writerows(data_to_write)
        
        data_to_write = [(f"{BASENAME}-Scene-{i+1:03d}", FPS, sum(si_values[scene[0]:scene[1]+1]), sum(si_values[scene[0]:scene[1]+1:1]), sum(si_values[scene[0]:scene[1]+1:3])) for i, scene in enumerate(scene_list)]
        output_path = os.path.join(output_folder, f"video_si.csv")

        with open(output_path, 'w', newline='') as csvfile:
            # Define the writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Scene_Number', 'FPS', 'SI_Value_0', 'SI_Value_1', 'SI_Value_2'])
            csv_writer.writerows(data_to_write)

    print(f"(SI data) {OUTPUT_DIR}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <VIDEO_PATH>")
        sys.exit(1)
    else:
        scenes_folder = "../output/timestamp"

        VIDEO_PATH = sys.argv[1]
        main(VIDEO_PATH, scenes_folder)