# calculate_metrics.py
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# 配置参数
affectnet_input_file = "./UniFER/eval_affectnet/results/affectnet_unifer_7b_results.json"
rafdb_input_file = "./UniFER/eval_rafdb/results/rafdb_unifer_7b_results.json"
ferplus_input_file = "./UniFER/eval_ferplus/results/ferplus_unifer_7b_results.json"
sfew_input_file = "./UniFER/eval_sfew_2.0/results/sfew_2.0_unifer_7b_results.json"
output_file = "./UniFER/eval_total/results/unifer_7b_results.json"

def extract_label_ferplus(response, candidate_labels):

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
        "angry": ["anger"],
        "contempt": ["scorn", "disdain"],
        "disgust": ["revulsion", "repulsion"],
        "fear": ["terror", "fright"],
        "happy": ["joy", "happiness"],
        "sad": ["sorrow", "unhappy"],
        "neutral": ["normal", "calm", "blank"],
        "surprise": ["shock", "astonishment"]
    }
    
    for label, alternatives in mapping.items():
        if any(alt in response_lower for alt in alternatives):
            return label
    
    return "unknown"

def extract_label_affectnet(response, candidate_labels):

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
        "anger": ["angry", "mad"],
        "contempt": ["scorn", "disdain"],
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

def extract_label_rafdb(response, candidate_labels):

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

def extract_label_sfew(response, candidate_labels):

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
        "angry": ["anger", "mad"],
        "disgust": ["revulsion", "repulsion"],
        "fear": ["terror", "fright"],
        "happy": ["joy", "happiness"],
        "sad": ["sorrow", "unhappy"],
        "neutral": ["normal", "calm", "blank"],
        "surprise": ["shock", "astonishment"]
    }
    
    for label, alternatives in mapping.items():
        if any(alt in response_lower for alt in alternatives):
            return label
    

    return "unknown"

def calculate_metrics(results_affectnet, results_rafdb, results_ferplus, results_sfew):
    true_labels = []
    pred_labels = []
    errors = 0
    unknowns = 0
    

    
    for item in results_affectnet:
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]
        model_response = item["model_response"]
        candidate_labels = item["candidate_labels"]
        
    
        pred_label = extract_label_affectnet(model_response, candidate_labels)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if pred_label == "unknown":
            unknowns += 1

    for item in results_rafdb:
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]
        model_response = item["model_response"]
        candidate_labels = item["candidate_labels"]
        
        pred_label = extract_label_rafdb(model_response, candidate_labels)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if pred_label == "unknown":
            unknowns += 1

    for item in results_ferplus:
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]
        model_response = item["model_response"]
        candidate_labels = item["candidate_labels"]
        
    
        pred_label = extract_label_ferplus(model_response, candidate_labels)

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if pred_label == "unknown":
            unknowns += 1
            
    for item in results_sfew:
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]
        model_response = item["model_response"]
        candidate_labels = item["candidate_labels"]
        
      
        pred_label = extract_label_sfew(model_response, candidate_labels)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        if pred_label == "unknown":
            unknowns += 1
    
    label_mapping = {'angry': 'anger', 'happy': 'happiness', 'sad': 'sadness'}
    true_labels = [label_mapping.get(item, item) for item in true_labels]
    pred_labels = [label_mapping.get(item, item) for item in pred_labels]
            
 
    print(f"\nTotal items: {len(results_affectnet)+len(results_sfew)+len(results_ferplus)+len(results_rafdb)}")
    print(f"Errors: {errors}")
    print(f"Unknown predictions: {unknowns}")
    print(f"Valid predictions: {len(true_labels)}")
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    
    
    candidate_labels = sorted(list(set(true_labels)))

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
    with open(affectnet_input_file, "r") as f:
        results_affectnet = json.load(f)
    
    with open(rafdb_input_file, "r") as f:
        results_rafdb = json.load(f)
        
    with open(ferplus_input_file, "r") as f:
        results_ferplus = json.load(f)

    with open(sfew_input_file, "r") as f:
        results_sfew = json.load(f)

    calculate_metrics(results_affectnet, results_rafdb, results_ferplus, results_sfew)