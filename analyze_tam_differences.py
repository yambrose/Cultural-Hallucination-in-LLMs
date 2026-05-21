import os
import json
import pandas as pd

langs = ['en', 'hi', 'ta']
base_dir = "model_outputs_tamil/precise_ta"

# Load golden datasets
excel_files = {
    'en': 'datasets/IndicWikiQA-Tam_EN.xlsx',
    'hi': 'datasets/IndicWikiQA-Tam_HI.xlsx',
    'ta': 'datasets/IndicWikiQA-Tam_TA.xlsx'
}

golden_data = {}
for lang, file_path in excel_files.items():
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Store as list of dicts for easy access by index
        golden_data[lang] = df.to_dict('records')

def analyze():
    for lang in langs:
        folder_path = os.path.join(base_dir, lang)
        if not os.path.exists(folder_path):
            continue
            
        gold = golden_data.get(lang, [])
        num_gold = len(gold)
        
        print(f"\n======================================")
        print(f"Language: {lang.upper()} (Total Gold Questions: {num_gold})")
        print(f"======================================")
        
        for file in os.listdir(folder_path):
            if not file.endswith('.jsonl'):
                continue
            
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                if line.strip():
                    data.append(json.loads(line))
            
            print(f"\n[{file}] - Total Questions: {len(data)}")
            
            # Find missing questions based on ID or index
            present_ids = set()
            differences = []
            
            for item in data:
                # Assuming the id matches the row in Excel
                # Let's try matching by id or just index mapping
                q_id = item.get('id')
                if q_id is not None:
                    present_ids.add(q_id)
                    
                    if q_id < len(gold):
                        gold_item = gold[q_id]
                        model_q = item.get('question', '').strip()
                        model_gold_a = str(item.get('gold_answer', '')).strip()
                        
                        gold_q = str(gold_item.get('Question', '')).strip()
                        excel_gold_a = str(gold_item.get('Answer', '')).strip()
                        
                        if model_q != gold_q or model_gold_a != excel_gold_a:
                            differences.append({
                                'id': q_id,
                                'model_q': model_q,
                                'gold_q': gold_q,
                                'model_a': model_gold_a,
                                'gold_a': excel_gold_a
                            })
            
            expected_ids = set(range(num_gold))
            missing_ids = expected_ids - present_ids
            
            if missing_ids:
                print(f"  -> Missing {len(missing_ids)} questions. IDs: {sorted(list(missing_ids))}")
            else:
                print(f"  -> No missing questions.")
                
            if differences:
                print(f"  -> Found {len(differences)} differences between JSONL and Excel Golden Data:")
                for diff in differences[:5]: # Show up to 5
                    print(f"     * ID {diff['id']}: ")
                    if diff['model_q'] != diff['gold_q']:
                        print(f"       JSONL Q: {diff['model_q']}\n       EXCEL Q: {diff['gold_q']}")
                    if diff['model_a'] != diff['gold_a']:
                        print(f"       JSONL Gold Ans: {diff['model_a']}\n       EXCEL Ans: {diff['gold_a']}")
                if len(differences) > 5:
                    print(f"     ... and {len(differences) - 5} more differences.")
            else:
                print(f"  -> No discrepancies in questions/golden answers.")

if __name__ == "__main__":
    analyze()
