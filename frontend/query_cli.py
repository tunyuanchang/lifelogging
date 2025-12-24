#!/usr/bin/env python3
"""
CLI-based semantic search using CLIP + FAISS.
Supports multiple text or image queries in a loop until "exit" is typed.
"""

import sys
import os
import re
import torch
import faiss
import pandas as pd
from PIL import Image
import open_clip

clipdata = []

# ---------------------------
# Search functions
# ---------------------------
def search(text_features, k):
    D, I = index.search(text_features, k)
    return D, I

def filter_and_label_results(I, D, results_per_page=10, selected_page=1):
    labels = get_labels()
    kfresults = []
    kfresultsidx = []
    kfscores = []
    num_results = len(I[0])

    if num_results == 0:
        return [], [], []

    ifrom = (selected_page - 1) * results_per_page
    ito = selected_page * results_per_page
    if ifrom >= num_results:
        return [], [], []

    for i in range(ifrom, min(ito, num_results)):
        idx = I[0][i]
        score = D[0][i]
        if idx == -1:
            continue
        kfresults.append(str(labels[idx]))
        kfresultsidx.append(int(idx))
        kfscores.append(float(score))
    return kfresults, kfresultsidx, kfscores

def get_labels():
    return labels

# ---------------------------
# CLI arguments
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <keyframe-base-root> <csv-file>")
        exit(1)

    keyframe_base_root = sys.argv[1]
    csv_file = sys.argv[2]
    top_k = 10  # default top K results

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ---------------------------
    # Device setup
    # ---------------------------

    # ---------------------------
    # Load CLIP model
    # ---------------------------
    modelname = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    match = re.search(r"openclip-[^-]+-(.+?)_(.+)\.csv", csv_file)
    if match:
        modelname = match.group(1)
        pretrained = match.group(2)
    print(f"Model: {modelname}, pretrained: {pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained, device=device)
    print("Model loaded")

    # ---------------------------
    # Load CLIP features from CSV
    # ---------------------------
    def load_clip_features(csvfilename):
        print(f'Loading features from {csvfilename}')
        csvdata = pd.read_csv(csvfilename, sep=",", header=None)
        data = csvdata.iloc[:, 1:].values.astype('float32')
        datalabels = csvdata.iloc[:, 0].tolist()
        clipdata.append(data)
        d = data.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine similarity
        index.add(data)
        print(f"FAISS index built: {index.ntotal} vectors")
        return index, datalabels

    index, labels = load_clip_features(csv_file)

    # ---------------------------
    # Main loop for multiple queries
    # ---------------------------
    print("\nEnter text or image file path to query. Type 'quit' to exit.\n")

    while True:
        query_input = input("Query> ").strip()
        if query_input.lower() in ["exit", "quit"]:
            print("Quit.")
            break
        if not query_input:
            continue

        with torch.no_grad():
            try:
                if query_input.lower().endswith((".png", ".jpg", ".jpeg")):
                    query_path = os.path.join(keyframe_base_root, query_input)
                    if os.path.isfile(query_path):
                    # Image query
                        print(f"Encoding image: {query_input}")
                        image = preprocess(Image.open(query_path)).unsqueeze(0).to(device)
                        features = model.encode_image(image).cpu().numpy()
                    else:
                        print(f"Error: Image file does not exist: {query_path}")

                else:
                    # Text query
                    tokens = open_clip.tokenize([query_input]).to(device)
                    features = model.encode_text(tokens).cpu().numpy()

                D, I = search(features, top_k)
                results, ids, scores = filter_and_label_results(I, D, results_per_page=top_k, selected_page=1)

                # print(f"\nTop {len(results)} results:")
                # for r, idx, score in zip(results, ids, scores):
                #     print(f"- ID: {idx}, Label: {r}, Score: {score:.4f}")
                # print("\n")
            except Exception as e:
                print(f"Error during search: {e}")

        