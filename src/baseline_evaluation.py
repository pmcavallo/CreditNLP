"""
Baseline Evaluation for CreditNLP
Tests Claude's ability to predict startup default risk using few-shot prompting.
This establishes the accuracy floor that fine-tuning needs to beat.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from tqdm import tqdm


def load_data(filepath: str) -> list[dict]:
    """Load JSONL data file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def stratified_split(data: list[dict], test_ratio: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Split data maintaining label distribution."""
    random.seed(seed)
    
    # Separate by label
    label_0 = [d for d in data if d['default_label'] == 0]
    label_1 = [d for d in data if d['default_label'] == 1]
    
    # Shuffle
    random.shuffle(label_0)
    random.shuffle(label_1)
    
    # Calculate split sizes
    test_size_0 = int(len(label_0) * test_ratio)
    test_size_1 = int(len(label_1) * test_ratio)
    
    # Split
    test_set = label_0[:test_size_0] + label_1[:test_size_1]
    train_set = label_0[test_size_0:] + label_1[test_size_1:]
    
    # Shuffle combined sets
    random.shuffle(test_set)
    random.shuffle(train_set)
    
    return train_set, test_set


def select_few_shot_examples(train_set: list[dict], n_per_class: int = 3) -> list[dict]:
    """Select examples with clearest signal patterns (highest absolute risk scores)."""
    label_0 = [d for d in train_set if d['default_label'] == 0]
    label_1 = [d for d in train_set if d['default_label'] == 1]
    
    # Sort by absolute risk score (most extreme examples)
    label_0_sorted = sorted(label_0, key=lambda x: x['risk_score'])  # Most negative first
    label_1_sorted = sorted(label_1, key=lambda x: x['risk_score'], reverse=True)  # Most positive first
    
    # Take top n from each
    examples = label_0_sorted[:n_per_class] + label_1_sorted[:n_per_class]
    
    # Interleave for balanced presentation
    result = []
    for i in range(n_per_class):
        result.append(label_0_sorted[i])
        result.append(label_1_sorted[i])
    
    return result


def build_prompt(few_shot_examples: list[dict], test_application: dict) -> str:
    """Construct the few-shot prompt."""
    prompt = """You are a credit risk analyst evaluating startup loan applications. Based on the application narrative, predict whether the startup will default on the loan.

Here are examples of past applications and their outcomes:

"""
    
    for i, example in enumerate(few_shot_examples, 1):
        outcome = "NO_DEFAULT" if example['default_label'] == 0 else "DEFAULT"
        prompt += f"EXAMPLE {i}:\n{example['application_text']}\nOutcome: {outcome}\n\n"
    
    prompt += f"""Now evaluate this new application:
{test_application['application_text']}

Respond with exactly one word: DEFAULT or NO_DEFAULT"""
    
    return prompt


def parse_prediction(response_text: str) -> int:
    """Parse model response to binary label."""
    response_clean = response_text.strip().upper()
    
    if "NO_DEFAULT" in response_clean or "NO DEFAULT" in response_clean:
        return 0
    elif "DEFAULT" in response_clean:
        return 1
    else:
        # Ambiguous response - default to majority class
        print(f"  Warning: Ambiguous response '{response_text}', defaulting to NO_DEFAULT")
        return 0


def calculate_metrics(predictions: list[int], actuals: list[int]) -> dict:
    """Calculate evaluation metrics."""
    # Confusion matrix components
    tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
    
    # Metrics
    accuracy = (tp + tn) / len(predictions) if predictions else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
    }


def estimate_cost(n_requests: int, input_tokens_per_request: int = 3000, output_tokens_per_request: int = 5) -> dict:
    """Estimate API cost using Sonnet pricing."""
    total_input = n_requests * input_tokens_per_request
    total_output = n_requests * output_tokens_per_request
    
    # Sonnet pricing: $3/M input, $15/M output
    input_cost = total_input * 3 / 1_000_000
    output_cost = total_output * 15 / 1_000_000
    
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(input_cost + output_cost, 4)
    }


def main():
    # Paths
    data_path = Path("data/synthetic_applications.jsonl")
    results_path = Path("results/baseline_evaluation.json")
    results_path.parent.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(data_path)
    print(f"  Loaded {len(data)} applications")
    
    # Split data
    print("\nSplitting data (80/20, stratified)...")
    train_set, test_set = stratified_split(data, test_ratio=0.2, seed=42)
    print(f"  Train: {len(train_set)} samples")
    print(f"  Test: {len(test_set)} samples")
    
    # Verify stratification
    train_default_rate = sum(1 for d in train_set if d['default_label'] == 1) / len(train_set)
    test_default_rate = sum(1 for d in test_set if d['default_label'] == 1) / len(test_set)
    print(f"  Train default rate: {train_default_rate:.1%}")
    print(f"  Test default rate: {test_default_rate:.1%}")
    
    # Select few-shot examples
    print("\nSelecting few-shot examples (3 per class, clearest signals)...")
    few_shot_examples = select_few_shot_examples(train_set, n_per_class=3)
    for ex in few_shot_examples:
        label = "DEFAULT" if ex['default_label'] == 1 else "NO_DEFAULT"
        print(f"  {ex['application_id']}: {label} (risk_score: {ex['risk_score']})")
    
    # Cost estimation
    print("\nCost estimation:")
    cost_estimate = estimate_cost(len(test_set))
    print(f"  Input tokens: ~{cost_estimate['total_input_tokens']:,}")
    print(f"  Output tokens: ~{cost_estimate['total_output_tokens']:,}")
    print(f"  Estimated cost: ${cost_estimate['total_cost_usd']:.2f}")
    
    # Initialize client
    print("\nInitializing Anthropic client...")
    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    # Evaluation loop
    print("\nRunning evaluation...")
    predictions = []
    actuals = []
    detailed_results = []
    
    for i, test_app in enumerate(tqdm(test_set, desc="Evaluating")):
        # Build prompt
        prompt = build_prompt(few_shot_examples, test_app)
        
        # Call API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        response_text = response.content[0].text
        prediction = parse_prediction(response_text)
        actual = test_app['default_label']
        
        predictions.append(prediction)
        actuals.append(actual)
        
        detailed_results.append({
            "application_id": test_app['application_id'],
            "actual": actual,
            "predicted": prediction,
            "correct": prediction == actual,
            "response_text": response_text,
            "risk_score": test_app['risk_score']
        })
        
        # Progress update every 25 samples
        if (i + 1) % 25 == 0:
            running_accuracy = sum(1 for r in detailed_results if r['correct']) / len(detailed_results)
            print(f"\n  Progress: {i+1}/{len(test_set)} | Running accuracy: {running_accuracy:.1%}")
    
    # Calculate final metrics
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS")
    print("="*50)
    
    metrics = calculate_metrics(predictions, actuals)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.1%}")
    print(f"\nDEFAULT Class Metrics:")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  F1 Score: {metrics['f1_score']:.1%}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"                 NO_DEF  DEFAULT")
    print(f"  Actual NO_DEF    {cm['true_negatives']:3d}     {cm['false_positives']:3d}")
    print(f"  Actual DEFAULT   {cm['false_negatives']:3d}     {cm['true_positives']:3d}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "claude-sonnet-4-20250514",
        "test_set_size": len(test_set),
        "few_shot_count": len(few_shot_examples),
        "metrics": metrics,
        "cost_estimate": cost_estimate,
        "few_shot_examples": [
            {
                "application_id": ex['application_id'],
                "default_label": ex['default_label'],
                "risk_score": ex['risk_score']
            }
            for ex in few_shot_examples
        ],
        "detailed_results": detailed_results
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
