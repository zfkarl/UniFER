# run_inference.py
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 配置参数
input_file = "./UniFER/eval_sfew_2.0/data/sfew_2.0_qa.json"
output_file = "./UniFER/eval_sfew_2.0/results/sfew_2.0_unifer_7b_results.json"
model_name = "./UniFER/model/UniFER-7B"

def load_model():
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        #torch_dtype=torch.bfloat16,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def perform_inference(data, model, processor):
    print("Starting inference...")
    results = []
    
    for item in tqdm(data, desc="Processing images"):
        try:
            # Prepare input
            image_path = item["image_path"]
            original_prompt = item["prompt"]
            new_prompt = original_prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "resized_height": 224, "resized_width": 224},
                        {"type": "text", "text": new_prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1024,
                    use_cache=True  
                )
            
            # Process output
            generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            results.append({
                **item,
                "model_response": response
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                **item,
                "model_response": f"ERROR: {str(e)}",
                "error": True
            })
    
    return results

if __name__ == "__main__":
    # Load dataset
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    # Load model
    model, processor = load_model()
    
    # Run inference
    results = perform_inference(dataset, model, processor)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference results saved to {output_file}")