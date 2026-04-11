import os
import pandas as pd
import numpy as np

# Configuration: Temperature range 0.9 to 1.08, step 0.02 (excluding 1.1)
# 0.9, 0.92, ..., 1.08 -> Total of 10 temperature points
TEMPERATURES = [f"{t:.2f}" for t in np.arange(0.9, 1.1, 0.02)]

# Base directory configuration
BASE_DIR = "results"
DATASET_NAME = "MATH-500" # Fixed as MATH-500

def calculate_metrics_for_problem(scores):
    """
    Calculate metrics for a single problem based on user-defined logic.
    scores: List of scores (0.0 or 1.0) for this problem across 10 temperatures.
    """
    # Pass@10: If at least one of the 10 samples is correct, count as 1 point.
    pass_at_10 = 1.0 if any(s == 1.0 for s in scores) else 0.0
    
    # Pass@1: For every correct answer among the 10, add 0.1 points (equivalent to calculating the average).
    pass_at_1 = sum(scores) * 0.1
    
    return pass_at_1, pass_at_10

def process_math500(model_path):
    """
    Logic specifically for processing MATH-500.
    1. Iterate through all temperature directories.
    2. Aggregate scores for each global_index.
    3. Calculate overall Pass@1 and Pass@10.
    """
    dataset_path = os.path.join(model_path, DATASET_NAME)
    if not os.path.isdir(dataset_path):
        return None

    # Dictionary to store score history for each problem: {global_index: [score_t1, score_t2, ...]}
    question_scores = {}

    # 1. Iterate through all temperature directories to collect data
    for temp in TEMPERATURES:
        temp_dir = os.path.join(dataset_path, temp)
        csv_path = os.path.join(temp_dir, "run.csv") # Only one run.csv per directory
        
        if not os.path.exists(csv_path):
            print(f"  [Warning] Missing {csv_path}")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                try:
                    idx = int(row['global_index'])
                    score = float(row['score'])
                    
                    if idx not in question_scores:
                        question_scores[idx] = []
                    question_scores[idx].append(score)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            print(f"  [Error] Failed to read {csv_path}: {e}")

    if not question_scores:
        return None

    # 2. Calculate metrics for each problem and accumulate
    total_pass1 = 0.0
    total_pass10 = 0.0
    valid_question_count = 0

    for idx, scores in question_scores.items():
        # Ensure the problem has gathered 10 samples (corresponding to 10 temperatures)
        # If some temperatures are missing resulting in fewer than 10, we can choose to skip or calculate proportionally.
        # Based on your description "collected 10 in total", we assume 10 are required for valid statistics here.
        if len(scores) == 10:
            p1, p10 = calculate_metrics_for_problem(scores)
            total_pass1 += p1
            total_pass10 += p10
            valid_question_count += 1


    if valid_question_count == 0:
        return None

    # Calculate average Pass@1 and Pass@10
    avg_pass1 = total_pass1 / valid_question_count
    avg_pass10 = total_pass10 / valid_question_count

    return {
        "Pass@1": round(avg_pass1, 3),
        "Pass@10": round(avg_pass10, 3),
        "Valid_Questions": valid_question_count
    }

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: '{BASE_DIR}' not found.")
        return

    summary = []

    print(f"Scanning models in '{BASE_DIR}' for dataset '{DATASET_NAME}'...")

    for model_name in sorted(os.listdir(BASE_DIR)):
        model_path = os.path.join(BASE_DIR, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"Processing model: {model_name}")
        
        metrics = process_math500(model_path)
        
        if metrics:
            summary.append({"Model": model_name, **metrics})
            print(f"  -> Pass@1: {metrics['Pass@1']}, Pass@10: {metrics['Pass@10']}")
        else:
            print(f"  -> No valid data found.")

    # Save results
    if summary:
        df = pd.DataFrame(summary)
        output_file = "ATS.csv"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
        print(df.to_string(index=False))
    else:
        print("\nNo valid data processed.")

if __name__ == "__main__":
    main()