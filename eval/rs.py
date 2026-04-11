import os
import pandas as pd
from collections import defaultdict

def compute_pass_at_k(temp_dir):
    """
    Read csv from temp_dir and calculate Pass@1 and Pass@10.
    """
    if not os.path.exists(temp_dir):
        return None

    question_scores = defaultdict(list)

    for i in range(1, 11):
        csv_path = os.path.join(temp_dir, f"run{i}.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Error] Failed to read {csv_path}: {e}")
            continue

        if 'global_index' not in df.columns or 'score' not in df.columns:
            print(f"[Warning] Missing required columns in {csv_path}")
            continue

        for _, row in df.iterrows():
            try:
                idx = int(row['global_index'])
                score = float(row['score'])
                question_scores[idx].append(score)
            except (ValueError, TypeError):
                continue  # Skip invalid values

    if not question_scores:
        return None

    # Pass@1: Proportion of correct answers across all samples
    all_scores = [s for scores in question_scores.values() for s in scores]
    pass_at_1 = sum(all_scores) / len(all_scores)

    # Pass@10: A question counts as passed if at least one attempt is correct
    pass_at_10_count = sum(
        1 for scores in question_scores.values()
        if any(s == 1.0 for s in scores)
    )
    pass_at_10 = pass_at_10_count / len(question_scores)

    return {
        'Pass@1': round(pass_at_1, 3),
        'Pass@10': round(pass_at_10, 3)
    }

def main():
    base_dir = "results"
    results = {}

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        results[model_name] = {}

        # Iterate through top-level dataset directories
        for dataset_name in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # === Standard datasets ===
            temp_dir = os.path.join(dataset_path, "0.8")
            metrics = compute_pass_at_k(temp_dir)
            if metrics is not None:
                results[model_name][dataset_name] = metrics

    # === Print results ===
    print("\n" + "=" * 70)
    print("Pass@K Results (Temperature = 0.8, RS Sampling)")
    print("=" * 70)
    for model in sorted(results.keys()):
        print(f"\nModel: {model}")
        for dataset in sorted(results[model].keys()):
            m = results[model][dataset]
            print(f"  - {dataset:<20} -> Pass@1 = {m['Pass@1']:<6} | Pass@10 = {m['Pass@10']}")

    # === Save to CSV ===
    rows = []
    for model, datasets in results.items():
        for dataset, metrics in datasets.items():
            rows.append({
                "Model": model,
                "Dataset": dataset,
                "Pass@1": metrics['Pass@1'],
                "Pass@10": metrics['Pass@10']
            })

    if rows:
        df_out = pd.DataFrame(rows)
        output_file = "RS.csv"
        
        # Ensure experiments directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
    else:
        print("\nNo valid data found.")

if __name__ == "__main__":
    main()