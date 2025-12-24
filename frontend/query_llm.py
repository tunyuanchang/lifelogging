#!/usr/bin/env python3
"""
CLI-based semantic search using CLIP + FAISS, enhanced with Multimodal RAG
using Qwen3-VL for answering questions based on retrieved content.

To run:
1. pip install transformers torch accelerate faiss-cpu pandas open-clip-torch Pillow
2. python3 <script_name.py> <keyframe-base-root> <csv-file>

To query and get an LLM answer, end your query with '?' or 'LLM'.
"""

import sys
import os
import re
import torch
import faiss
import pandas as pd
from PIL import Image
import open_clip

# Added imports for Qwen3-VL
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Global variables
clipdata = []
index = None
labels = []
llm_model = None
llm_processor = None
keyframe_base_root = "" # Will be set in main


# ---------------------------
# Search functions
# ---------------------------
def search(text_features, k):
    """Performs search against the global FAISS index."""
    if index is None:
        raise RuntimeError("FAISS index is not initialized.")
    # Ensure the features are in the correct shape/type for FAISS
    features_np = text_features.astype('float32').reshape(1, -1)
    D, I = index.search(features_np, k)
    return D, I

def filter_and_label_results(I, D, results_per_page=10, selected_page=1):
    """Extracts labels and scores from search results."""
    # Note: labels is retrieved via the global scope or a getter (as per original code structure)
    global labels
    
    kfresults = []
    kfresultsidx = []
    kfscores = []
    
    # Check if I and D contain data
    if len(I) == 0 or len(I[0]) == 0:
        return [], [], []
        
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
        # Ensure idx is within the bounds of the labels list
        if 0 <= idx < len(labels):
            kfresults.append(str(labels[idx])) # Keyframe file path/label
            kfresultsidx.append(int(idx))
            kfscores.append(float(score))
        else:
            print(f"Warning: Index {idx} out of bounds for labels list (size {len(labels)}).")

    return kfresults, kfresultsidx, kfscores


# ---------------------------
# RAG Function using Qwen3-VL
# ---------------------------
def answer_with_llm(query_input, keyframe_paths, keyframe_base_root):
    """
    Constructs a multimodal prompt with the user query and retrieved images/paths,
    and generates an answer using the Qwen3-VL model.
    """
    global llm_model, llm_processor
    
    if llm_model is None or llm_processor is None:
        return "LLM model or processor is not loaded. Cannot generate RAG answer."

    # 1. Prepare the user's message parts (Images and Text)
    messages = []
    
    # Check if the original query was an image path that exists
    is_image_query_input = query_input.lower().endswith((".png", ".jpg", ".jpeg"))
    if is_image_query_input:
        query_path = os.path.join(keyframe_base_root, query_input)
        if os.path.isfile(query_path):
             # Include the original query image if it exists
            print(f"Including query image: {query_path}")
            messages.append({"type": "image", "image": Image.open(query_path).convert("RGB")})

    # 2. Add the retrieved keyframes/images as context (RAG context)
    retrieved_images = []
    for kf_label in keyframe_paths:
        # Assuming the label is the relative path/filename
        kf_path = os.path.join(keyframe_base_root, kf_label) 
        if os.path.isfile(kf_path):
            try:
                retrieved_images.append(Image.open(kf_path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Could not load retrieved image {kf_path}: {e}")

    # Add retrieved images to the message
    for i, img in enumerate(retrieved_images):
        messages.append({"type": "image", "image": img})
        # print(f"Included retrieved image {i+1}.") # Debug

    # 3. Formulate the RAG prompt text 
    if retrieved_images:
        context_text = "Based on the images provided above, answer the following question: "
        # If the input query was an image, use a descriptive question if possible, or just the filename as context.
        # For simplicity, we use the original (potentially cleaned) text query as the prompt.
        messages.append({"type": "text", "text": context_text + query_input})
    else:
        # Fallback if no images were retrieved (just a direct LLM query)
        messages.append({"type": "text", "text": query_input})


    # 4. Process and generate the answer
    try:
        # Apply chat template and tokenize
        # We wrap the messages in a single 'user' role for the instruction format
        inputs = llm_processor.apply_chat_template(
            [{"role": "user", "content": messages}],
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        # Move inputs to the LLM's device
        inputs = inputs.to(llm_model.device)

        # Inference: Generation of the output
        generated_ids = llm_model.generate(
            **inputs, 
            max_new_tokens=512, 
            pad_token_id=llm_processor.tokenizer.eos_token_id 
        )
        
        # Decode the generated text, excluding the input prompt part
        input_len = inputs['input_ids'].shape[1]
        response_text = llm_processor.decode(generated_ids.tolist()[0][input_len:], skip_special_tokens=True).strip()
        
        return response_text

    except Exception as e:
        return f"Error during LLM generation: {e}"


# ---------------------------
# CLI arguments and main execution
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <keyframe-base-root> <csv-file>")
        exit(1)

    keyframe_base_root = sys.argv[1] # Set global keyframe_base_root
    csv_file = sys.argv[2]
    top_k = 10  # default top K results

    # Device Setup (Use GPU if available for both CLIP and LLM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ---------------------------
    # Load CLIP model
    # ---------------------------
    modelname = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    # Attempt to extract model name from CSV filename
    match = re.search(r"openclip-[^-]+-(.+?)_(.+)\.csv", csv_file)
    if match:
        modelname = match.group(1)
        pretrained = match.group(2)
    print(f"CLIP Model: {modelname}, pretrained: {pretrained}")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained, device=device)
        model.to(device)
        print("CLIP Model loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        exit(1)

    # ---------------------------
    # Load CLIP features from CSV and build FAISS index
    # ---------------------------
    def load_clip_features(csvfilename):
        global index, labels
        print(f'Loading features from {csvfilename}')
        try:
            csvdata = pd.read_csv(csvfilename, sep=",", header=None)
            # FAISS requires numpy float32
            data = csvdata.iloc[:, 1:].values.astype('float32') 
            labels = csvdata.iloc[:, 0].tolist()
            # Normalize vectors for Inner Product (IP) to be equivalent to Cosine Similarity
            data = data / torch.from_numpy(data).norm(p=2, dim=-1, keepdim=True).numpy()
            
            d = data.shape[1]
            index = faiss.IndexFlatIP(d)  # cosine similarity via Inner Product on normalized vectors
            index.add(data)
            print(f"FAISS index built: {index.ntotal} vectors of dimension {d}")
            return index, labels
        except Exception as e:
            print(f"Error loading features/building FAISS index: {e}")
            exit(1)

    load_clip_features(csv_file)
    # Global 'index' and 'labels' are now set

    # ---------------------------
    # Load Qwen3-VL LLM
    # ---------------------------
    llm_model_name = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"\nLoading LLM model: {llm_model_name}")
    try:
        # Load the LLM model and processor
        llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            llm_model_name, 
            dtype="auto", 
            device_map="auto" # Auto-map to available resources (e.g., GPU)
        ).eval()
        llm_processor = AutoProcessor.from_pretrained(llm_model_name)
        print("Qwen3-VL-2B-Instruct model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Qwen3-VL-2B-Instruct. Check your installation and GPU memory. LLM functionality disabled. Error: {e}")
        llm_model = None
        llm_processor = None


    # ---------------------------
    # Main loop for multiple queries
    # ---------------------------
    print("\n" + "="*50)
    print("Multimodal RAG CLI Ready.")
    print("Enter text or image file path to query.")
    # print("To trigger RAG for an answer, end your query with '?' or 'LLM'.")
    print("Type 'quit' or 'exit' to quit.")
    print("="*10 + "\n")

    while True:
        query_input = input("Query> ").strip()
        if query_input.lower() in ["exit", "quit"]:
            print("Quit.")
            break
        if not query_input:
            continue

        # Determine if it's a RAG query
        is_rag_query = query_input.endswith('?') or query_input.lower().endswith(" llm")

        # Clean the input for CLIP search (used as the prompt for CLIP and the LLM)
        clip_query = query_input.replace('?', '').replace(' llm', '').strip() if is_rag_query else query_input
        if not clip_query:
            print("Query is empty. Please provide a query.")
            continue
        
        # ---------------------------
        # CLIP Search
        # ---------------------------
        with torch.no_grad():
            try:
                features = None
                
                # 1. Determine Input Type and Encode
                is_image_query = clip_query.lower().endswith((".png", ".jpg", ".jpeg"))
                if is_image_query:
                    query_path = os.path.join(keyframe_base_root, clip_query)
                    if os.path.isfile(query_path):
                        # Image query
                        print(f"Encoding image for search: {clip_query}")
                        image = preprocess(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
                        features = model.encode_image(image).cpu().numpy().astype('float32')
                    else:
                        print(f"Error: Image file does not exist: {query_path}. Treating as text query.")
                        tokens = open_clip.tokenize([clip_query]).to(device)
                        features = model.encode_text(tokens).cpu().numpy().astype('float32')
                else:
                    # Text query
                    tokens = open_clip.tokenize([clip_query]).to(device)
                    features = model.encode_text(tokens).cpu().numpy().astype('float32')
                
                # 2. Normalize features (already done in load_clip_features for the index, 
                # but good practice to ensure query vector is also normalized)
                features = features / torch.from_numpy(features).norm(p=2, dim=-1, keepdim=True).numpy()

                # 3. Perform Search
                D, I = search(features, top_k)
                results, ids, scores = filter_and_label_results(I, D, results_per_page=top_k, selected_page=1)

                # 4. Print Semantic Search Results
                print(f"\n--- CLIP Semantic Search (Top {len(results)}) ---")
                for r, idx, score in zip(results, ids, scores):
                    print(f"- Label: {r}, Similarity score: {score:.4f}")
                print("------------------------------------------")

                # ---------------------------
                # RAG Step (If requested)
                # ---------------------------
                if is_rag_query:
                    print("\n--- RAG with Qwen3-VL ---")
                    retrieved_keyframe_paths = results # The labels are the keyframe paths/filenames
                    
                    # Pass the original query (potentially an image path) and retrieved paths
                    llm_answer = answer_with_llm(
                        clip_query, 
                        retrieved_keyframe_paths, 
                        keyframe_base_root
                    )
                    
                    print(f"LLM Answer:\n{llm_answer}\n")
                    print("--------------------------\n")
            
            except Exception as e:
                print(f"Error during search: {e}")