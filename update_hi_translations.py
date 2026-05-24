import os
import json
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator

base_dir = "model_outputs_tamil/precise_ta"
ignore_files = ["qwen-mt-plus.jsonl", "llama-4-maverick.jsonl"]

def translate_text(text):
    if not text or str(text).strip() == "":
        return ""
    try:
        return GoogleTranslator(source='en', target='hi').translate(str(text))
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

def main():
    en_dir = os.path.join(base_dir, "en")
    hi_dir = os.path.join(base_dir, "hi")
    
    models = [f for f in os.listdir(en_dir) if f.endswith(".jsonl") and f not in ignore_files]
    
    for model in models:
        en_path = os.path.join(en_dir, model)
        hi_path = os.path.join(hi_dir, model)
        
        if not os.path.exists(hi_path):
            continue
            
        print(f"Loading '{model}' and translating 222 items to Hindi...")
        with open(en_path, "r", encoding="utf-8") as f:
            en_data = [json.loads(line) for line in f if line.strip()]
            
        # Extract all english answers
        en_answers = [item.get("model_answer", "") for item in en_data]
        
        # Translate all using threading to speed up Google API requests slightly 
        # (Worker count is kept low to avoid free-tier API rate limits)
        with ThreadPoolExecutor(max_workers=4) as executor:
            hi_answers = list(executor.map(translate_text, en_answers))
            
        # Match with hi_data and overwrite
        with open(hi_path, "r", encoding="utf-8") as f:
            hi_data = [json.loads(line) for line in f if line.strip()]
            
        for i, item in enumerate(hi_data):
            q_id = item.get("id")
            
            # Find the corresponding english generation by ID
            matched = False
            for j, e_item in enumerate(en_data):
                if e_item.get("id") == q_id:
                    item["model_answer"] = hi_answers[j]
                    matched = True
                    break
                    
            if not matched:  # Fallback to pure index
                item["model_answer"] = hi_answers[i] if i < len(hi_answers) else ""
                
        with open(hi_path, "w", encoding="utf-8") as f:
            for item in hi_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"  -> Successfully retranslated and saved standard Hindi answers for {model}")

    print("\nFinished converting all English precise_ta model answers across the 6 models to Hindi!")

if __name__ == "__main__":
    main()
