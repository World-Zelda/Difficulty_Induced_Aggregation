import os
import pandas as pd
import numpy as np

TEMPERATURES = [f"{t:.1f}" for t in np.arange(0.2, 1.2, 0.1)]

def load_all_scores(base_path):
    """
    Load run.csv from all 10 temperature folders.
    Returns a dict: {global_index: [score_0, score_1, ..., score_9]}
    Each score is float (0.0 or 1.0 typically).
    """
    problem_to_scores = {}
    for temp in TEMPERATURES:
        csv_path = os.path.join(base_path, temp, "run.csv")
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None
        if 'global_index' not in df.columns or 'score' not in df.columns:
            return None
        
        for _, row in df.iterrows():
            gid = row['global_index']
            score = float(row['score'])
            if gid not in problem_to_scores:
                problem_to_scores[gid] = []
            problem_to_scores[gid].append(score)
    
    # Validate: each problem must have exactly 10 scores
    for gid, scores in problem_to_scores.items():
        if len(scores) != 10:
            return None
    
    return problem_to_scores

def compute_custom_pass(problem_scores):
    """
    Compute metrics according to user's custom definition:
    - Pass@1: for each problem, sum(score_i * 0.1) over 10 samples -> equivalent to mean(score)
    """
    pass1_total = 0.0
    num_problems = len(problem_scores)

    for scores in problem_scores.values():
        # Pass@1: each correct sample contributes 0.1
        correct_count = sum(1.0 for s in scores if s >= 0.99)
        pass1_total += correct_count * 0.1

    pass1 = round(pass1_total / num_problems, 3)

    return {"Pass@1": pass1}

def process_dataset(model_path, dataset_name):
    dataset_path = os.path.join(model_path, dataset_name)
    if not os.path.isdir(dataset_path):
        return None

    data = load_all_scores(dataset_path)
    if data is None:
        return None
    return compute_custom_pass(data)

def main():
    base_dir = "results"
    if not os.path.exists(base_dir):
        print(f"Error: '{base_dir}' not found.")
        return

    summary = []

    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"Processing model: {model_name}")

        # Other datasets
        for dataset_name in sorted(os.listdir(model_path)):
            metrics = process_dataset(model_path, dataset_name)
            if metrics:
                summary.append({"Model": model_name, "Dataset": dataset_name, **metrics})

    if summary:
        df = pd.DataFrame(summary)
        output_file = "DTS.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
        print(df.to_string(index=False))
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    main()