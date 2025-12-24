import csv
import argparse
import numpy as np
import sys
import os

# Fixed lists for resolution and quality variants
# The k-th resolution is paired with the k-th quality.
FIXED_RESOLUTIONS = [1.0, 4.0, 9.0, 16.0, 25.0]
FIXED_QUALITIES = [1.0, 0.9, 0.8, 0.7, 0.6]
MAX_RESOLUTION = max(FIXED_RESOLUTIONS)

def dp_solver(csv_file_path, capacity):
    """
    Solves a 0/1 Knapsack-like problem where each image must be assigned 
    one variant from the FIXED_RESOLUTIONS/QUALITIES lists.

    Args:
        csv_file_path (str): Path to the input CSV file (img_path, current_res, current_qual, original_score).
        capacity (float): The maximum allowed total "weight" (Resolution * Quality).

    Returns:
        tuple: (max_total_score, selected_variants).
    """
    
    # 1. Read and preprocess the data
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Skip header if present
            next(reader, None) 
            raw_data = list(reader)
    except FileNotFoundError:
        print(f"Error: File not found at **{csv_file_path}**")
        return 0, []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0, []

    # select all
    if capacity == -1:
        selections = []

        for row in raw_data:
            selections.append((row[0], float(row[1]), float(row[2])))

        return -1, -1, selections

    groups = []
    
    # 2. Pre-calculate all possible varients
    for row in raw_data:
        try:
            # [img path, current resolution, current quality, original score]
            img_path = row[0]
            cur_res = float(row[1])
            cur_qual = float(row[2])
            origin_score = float(row[3])
            
            current_group = []
            
            # Generate all possible variants for this image (the items in the group)
            for res, qual in zip(FIXED_RESOLUTIONS, FIXED_QUALITIES):
                
                # Skip this variant as it exceeds the image's current state
                if res > cur_res or qual > cur_qual:
                    continue
                
                # Calculate Weight and Score for this combination
                weight = res * qual
                int_weight = int(weight) # Cast to integer as required
                
                # Score formula: original score * (resolution * quality / max_resolution)
                score = origin_score * (weight / MAX_RESOLUTION)
                
                # Item data structure: (path, weight_int, score, res, qual)
                current_group.append({
                    'path': img_path,
                    'weight_int': int_weight,
                    'score': score,
                    'res': res,
                    'qual': qual
                })
                     
            
            if current_group:
                groups.append(current_group)
                 
        except Exception as e:
            print(f"Skipping malformed row: {row}. Error: {e}")
            continue

    # 3. DP: 0/1 Knapsack Problem
    
    num_groups = len(groups)
    
    dp = np.zeros(capacity + 1, dtype=float) # dp[w] = max score for weight w
    choice_index = {} # Maps (group_index, capacity_w) -> chosen_item_index_within_group
    
    for g in range(1, num_groups + 1):
        group = groups[g - 1]
        
        for w in range(capacity, 0, -1):
            max_score_for_w = dp[w] 
            best_variant_index = -1 
            
            # Iterate through all valid items (variants) within the current group
            for item_index, item in enumerate(group):
                int_weight = item['weight_int']
                score = item['score']
                
                if int_weight <= w:
                    score_if_chosen = score + dp[w - int_weight]
                    
                    if score_if_chosen > max_score_for_w:
                        max_score_for_w = score_if_chosen
                        best_variant_index = item_index
            
            dp[w] = max_score_for_w
            choice_index[(g, w)] = best_variant_index

    max_total_score = dp[capacity]

    # 4. Traceback
    selected_variants = []
    w = capacity
    
    for g in range(num_groups, 0, -1):
        chosen_index = choice_index.get((g, w), -1)
        
        if chosen_index != -1:
            chosen_item = groups[g - 1][chosen_index]
            
            selected_variants.append((
                chosen_item['path'], 
                chosen_item['res'], 
                chosen_item['qual']
            ))
            
            w -= chosen_item['weight_int'] 

    max_total_weight = capacity - w
    return max_total_score, max_total_weight, selected_variants[::-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add the required arguments
    parser.add_argument("--input", "-i", type=str, help="csv file of history content")
    parser.add_argument("--capacity", "-c", type=int, help="maximum capacity")
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    print("Running Knapsack Solver with Fixed Variants...")
    # print(f"**Resolution Variants (R):** {FIXED_RESOLUTIONS}")
    # print(f"**Quality Variants (Q):** {FIXED_QUALITIES}")
    # print(f"CSV Path: {args.input}")
    # print(f"Capacity: {args.capacity}")
    
    # Perform the calculation
    max_score, max_weight, selections = dp_solver(args.input, args.capacity)

    print("\n--- Results ---")
    # print(f"**Max Total Score Achieved: {max_score:.2f}**")
    # print(f"**Max Total Weight Achieved: {max_weight:.2f}**")

    print(f"Total Selected Items: {len(selections)}")
    
    if selections:
        # print("\n**Selected Image Variants:**")
        # print("Image Path | Resolution | Quality")
        # print("---|---|---")
        # for path, res, qual in selections:
        #     print(f"{path} | {res:.1f} | {qual:.1f}")

        output_file = "final_selection.csv"
        with open(output_file, 'w', newline='') as csvfile:
            # Define the writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(selections)

        print(f"(Final selection) {os.path.abspath(output_file)}")

    else:
        print("No valid items were selected or the capacity was too small.")