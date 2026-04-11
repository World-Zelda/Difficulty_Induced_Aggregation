import os
import pandas as pd
import numpy as np
from itertools import product
from collections import Counter

# === Configuration ===
MODELS = [
    'Qwen3-0.6B'
]

THETA_VALUES = [x / 100 for x in range(30, 61, 5)]
BETA_VALUES = [x / 100 for x in range(50, 151, 25)]

# Base directory
BASE_DIR = 'results'

# Range: 0.9 to 1.08, step 0.02 (Total 10 temperatures)
MATH_TEMPS = [f"{t:.2f}" for t in np.arange(0.9, 1.1, 0.02)]

# Define "Low Temperature" subset for C_low calculation
# We take the first 5 lower temperatures: 0.9, 0.92, 0.94, 0.96, 0.98
LOW_TEMP_SET = set(MATH_TEMPS[:5]) 


def weighted_aggregation(answers, temperatures, beta):
    """Weighted voting for all answers based on exp(beta * temperature)"""
    temp_vals = [float(t) for t in temperatures]
    weights = [np.exp(beta * t) for t in temp_vals]
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    
    score = {}
    for ans, w in zip(answers, norm_weights):
        score[ans] = score.get(ans, 0.0) + w
    
    return max(score, key=score.get) if score else ""


def evaluate_math500_induce(model, theta, beta):
    """Evaluate a single model on MATH-500 using the induce strategy"""
    print(f"\n[+] Evaluating {model} on MATH-500 with θ={theta}, β={beta}")
    
    all_answers = {}          # qid -> list of answers
    all_temperatures = {}     # qid -> list of temps
    ground_truth = {}         # qid -> correct answer
    
    found_any = False
    
    math_base = os.path.join(BASE_DIR, model, "MATH-500")
    if not os.path.exists(math_base):
        print(f"Directory not found: {math_base}")
        return None, None

    for temp in MATH_TEMPS:
        temp_dir = os.path.join(math_base, temp)
        if not os.path.exists(temp_dir):
            continue
        
        # Read the single run.csv in each temperature directory
        csv_path = os.path.join(temp_dir, "run.csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(
                csv_path,
                dtype={'extracted_answer': str},
                keep_default_na=False,
                na_values=['None', '', 'nan']
            )
        except Exception as e:
            print(f"Read error: {csv_path} - {e}")
            continue
        
        found_any = True
        for _, row in df.iterrows():
            # --- Get Question ID ---
            qid = None
            if 'question_id' in row and pd.notna(row['question_id']):
                qid_str = str(row['question_id']).strip()
                if qid_str not in ('', 'nan', 'None'):
                    qid = qid_str
            elif 'global_index' in row and pd.notna(row['global_index']):
                try:
                    qid = str(int(float(row['global_index'])))
                except (ValueError, TypeError):
                    pass
            
            if qid is None:
                continue
            
            # --- Extract Prediction and Ground Truth ---
            pred = row.get('extracted_answer', '')
            true = row.get('target_answer', '') or row.get('correct_answer', '')
            
            extracted = '' if (pd.isna(pred) or str(pred).strip() in ('', 'None', 'nan')) else str(pred).strip()
            true_clean = '' if (pd.isna(true) or str(true).strip() in ('', 'None', 'nan')) else str(true).strip()
            
            if qid not in all_answers:
                all_answers[qid] = []
                all_temperatures[qid] = []
                ground_truth[qid] = true_clean
            
            all_answers[qid].append(extracted)
            all_temperatures[qid].append(temp)
    
    if not found_any:
        print(f"No data found for {model}")
        return None, None
    
    predictions = {}
    for qid in all_answers:
        answers = all_answers[qid]
        temps = all_temperatures[qid]
        
        # We expect exactly 10 samples (one for each temperature)
        if len(answers) != 10:
            print(f"QID {qid}: only {len(answers)} samples (expected 10), skipped.")
            continue
        
        # --- Extract Low Temperature Group ---
        # Filter answers that belong to the LOW_TEMP_SET (first 5 temps)
        low_answers = [
            ans for ans, t in zip(answers, temps) if t in LOW_TEMP_SET
        ]
        
        # We expect 5 low-temp samples
        if len(low_answers) != 5:
            print(f"QID {qid}: low-temp samples = {len(low_answers)} (expected 5), skipped.")
            continue
        
        # --- Calculate C_low ---
        counter_low = Counter(low_answers)
        max_count = max(counter_low.values()) if counter_low else 0
        C_low = max_count / 5.0  # Denominator is 5 now
        
        # --- Two-stage Decision ---
        if C_low > theta:
            final_ans = counter_low.most_common(1)[0][0]
        else:
            final_ans = weighted_aggregation(answers, temps, beta)
        
        predictions[qid] = final_ans
    
    # --- Calculate Accuracy ---
    correct = 0
    total = 0
    for qid, pred in predictions.items():
        if qid in ground_truth:
            if pred.strip() == ground_truth[qid].strip():
                correct += 1
            total += 1
    
    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    return acc, total


def main():
    results = []
    
    for model in MODELS:
        best_acc = 0.0
        best_theta, best_beta = None, None
        
        print(f"\n{'='*60}")
        print(f"Processing {model} on MATH-500")
        print(f"{'='*60}")
        
        for theta, beta in product(THETA_VALUES, BETA_VALUES):
            acc, n = evaluate_math500_induce(model, theta, beta)
            if acc is not None:
                results.append({
                    'model': model,
                    'theta': theta,
                    'beta': beta,
                    'accuracy': round(acc, 4),
                    'total_problems': n
                })
                if acc > best_acc:
                    best_acc = acc
                    best_theta = theta
                    best_beta = beta
        
        if best_acc > 0:
            print(f"\nBest for {model}: {best_acc:.4f} (θ={best_theta}, β={best_beta})")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = "D_induce_results.csv"
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()