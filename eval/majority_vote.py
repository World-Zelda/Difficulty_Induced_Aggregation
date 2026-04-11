import os
import pandas as pd
from collections import Counter

# Generate temperature list: [0.90, 0.92, ..., 1.08] (left-closed, right-open interval [0.9, 1.1) with step 0.02)
TEMPERATURES = [f"{t:.2f}" for t in [0.9 + i * 0.02 for i in range(10)]]  # 10 temperatures total


def evaluate_majority_voting(model):
    print(f"\n[+] Evaluating {model} on MATH-500 (Majority Voting over 10 samples per problem)")

    dataset_base = os.path.join("results", model, "MATH-500")
    if not os.path.exists(dataset_base):
        print(f"Path not found: {dataset_base}")
        return None, None

    all_answers = {}      # qid -> list of predicted answers (expected length: 10)
    ground_truth = {}     # qid -> correct answer string

    # Iterate over each temperature directory
    for temp in TEMPERATURES:
        csv_path = os.path.join(dataset_base, temp, "run.csv")
        if not os.path.exists(csv_path):
            print(f"Missing file: {csv_path}, skipping temperature {temp}")
            continue

        try:
            df = pd.read_csv(
                csv_path,
                dtype={'extracted_answer': str},
                keep_default_na=False,
                na_values=['None', '', 'nan']
            )
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        for _, row in df.iterrows():
            # Safely extract question identifier
            qid = None
            if 'question_id' in row and pd.notna(row['question_id']) and str(row['question_id']).strip() not in ('', 'nan', 'None'):
                qid = str(row['question_id']).strip()
            elif 'global_index' in row and pd.notna(row['global_index']):
                try:
                    idx_int = int(float(row['global_index']))
                    qid = str(idx_int)
                except (ValueError, TypeError):
                    pass  # Invalid index; skip this row

            if qid is None:
                print(f"Skipping row with invalid question_id/global_index: {row.to_dict()}")
                continue

            # Extract predicted answer
            pred_ans = row.get('extracted_answer', '')
            extracted = '' if (pd.isna(pred_ans) or str(pred_ans).strip() in ('', 'None', 'nan')) else str(pred_ans).strip()

            # Extract ground truth answer (prefer 'target_answer', fallback to 'correct_answer')
            true_ans = row.get('target_answer', '') or row.get('correct_answer', '')
            true_clean = '' if (pd.isna(true_ans) or str(true_ans).strip() in ('', 'None', 'nan')) else str(true_ans).strip()

            # Initialize storage for new questions
            if qid not in all_answers:
                all_answers[qid] = []
                ground_truth[qid] = true_clean

            all_answers[qid].append(extracted)

    # Evaluate only problems with exactly 10 samples (one per temperature)
    correct = 0
    total = 0

    for qid, answers in all_answers.items():
        if len(answers) != 10:
            print(f"Problem {qid}: only {len(answers)} samples (expected 10), skipped.")
            continue

        total += 1
        # Perform majority voting: select the most frequent answer
        most_common_ans, _ = Counter(answers).most_common(1)[0]
        if most_common_ans.strip() == ground_truth[qid].strip():
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Majority Voting Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, total


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return

    # Auto-discover models: all subdirectories in 'results/'
    models = [
        name for name in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, name))
    ]

    if not models:
        print("No models found in 'results/' directory.")
        return

    print(f"Found {len(models)} models: {models}")

    results = []

    for model in models:
        acc, n = evaluate_majority_voting(model)
        if acc is not None:
            results.append({
                'model': model,
                'dataset': 'MATH-500',
                'majority_voting_accuracy': round(acc, 4),
                'total_problems': n
            })

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = "majority_voting_results.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print("\n=== Summary ===")
        for r in results:
            print(f"{r['model']}: {r['majority_voting_accuracy']:.4f} ({r['total_problems']} problems)")
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    main()