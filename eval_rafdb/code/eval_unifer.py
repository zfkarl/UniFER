# calculate_metrics.py
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


input_file = "./UniFER/eval_rafdb/results/rafdb_unifer_7b_results.json"
output_file = "./UniFER/eval_rafdb/results/rafdb_unifer_7b_metrics.json"

def extract_label(response, candidate_labels):

    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE)
    
    if answer_match:
        answer_text = answer_match.group(1).strip().lower()
     
        for label in candidate_labels:
            if label.lower() == answer_text.lower():
                return label

            if label.lower() in answer_text.lower():
                return label
    

    response_lower = response.lower()
    for label in candidate_labels:

        if re.search(rf'\b{label}\b', response_lower):
            return label
        

        if label in response_lower:
            return label
    
  
    mapping = {
        "anger": ["angry"],
        "disgust": ["revulsion", "repulsion"],
        "fear": ["terror", "fright"],
        "happiness": ["joy", "happy"],
        "sadness": ["sad"],
        "neutral": ["normal", "calm", "blank"],
        "surprise": ["shock", "astonishment"]
    }
    
    for label, alternatives in mapping.items():
        if any(alt in response_lower for alt in alternatives):
            return label
    
 
    return "unknown"

def calculate_metrics(results):
    true_labels = []
    pred_labels = []
    errors = 0
    unknowns = 0
    
    for item in results:
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]
        model_response = item["model_response"]
        candidate_labels = item["candidate_labels"]
        
     
        pred_label = extract_label(model_response, candidate_labels)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if pred_label == "unknown":
            unknowns += 1
    

    print(f"\nTotal items: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Unknown predictions: {unknowns}")
    print(f"Valid predictions: {len(true_labels)}")
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    

    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=candidate_labels)
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        labels=candidate_labels,
        zero_division=0,
        target_names=candidate_labels,
        output_dict=True
    )
    

    print("\n===== Overall Performance Metrics =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    

    print("\n===== Confusion Matrix =====")
    print("True Label (Rows) vs Predicted Label (Columns)")
    print(f"{'':>10}", end="")
    for label in candidate_labels:
        print(f"{label[:5]:>8}", end="")
    print("\n" + "-" * (10 + len(candidate_labels)*8))
    
    for i, true_label in enumerate(candidate_labels):
        print(f"{true_label[:10]:>10}", end="")
        for j, pred_label in enumerate(candidate_labels):
            print(f"{conf_matrix[i][j]:>8}", end="")
        print()
    

    print("\n===== Per-Class Performance Metrics =====")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    

    per_class_stats = []
    
    for emotion in candidate_labels:
        stats = class_report[emotion]
        precision_c = stats['precision']
        recall_c = stats['recall']
        f1_c = stats['f1-score']
        support_c = stats['support']
        
        per_class_stats.append({
            "emotion": emotion,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "support": support_c
        })
        
        print(f"{emotion:<12} {precision_c:<10.4f} {recall_c:<10.4f} {f1_c:<10.4f} {support_c:<10.0f}")
    

    sorted_stats = sorted(per_class_stats, key=lambda x: x['f1'], reverse=True)
    
    print("\n===== Classes Ranked by F1-Score =====")
    print(f"{'Rank':<5} {'Class':<12} {'F1-Score':<10}")
    for rank, stats in enumerate(sorted_stats, 1):
        print(f"{rank:<5} {stats['emotion']:<12} {stats['f1']:<10.4f}")
    

    detailed_results = {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "per_class": per_class_stats,
        "confusion_matrix": conf_matrix.tolist()
    }
    
 
    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print("\nPer-class metrics saved to per_class_metrics.json")

if __name__ == "__main__":
    with open(input_file, "r") as f:
        results = json.load(f)
    
    calculate_metrics(results)