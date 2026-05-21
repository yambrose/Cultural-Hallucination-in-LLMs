import os
import json

judge_dir = "judge_results_new"

def load_domains(in_dir):
    """
    Extracts the domain mappings from original .jsonl files.
    All models over the same split share the same IDs and domains, so processing one file is sufficient.
    """
    domains = {}
    for file in os.listdir(in_dir):
        if file.endswith(".jsonl"):
            path = os.path.join(in_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    q_id = data.get("id")
                    domain = data.get("domain", "Unknown")
                    if q_id is not None:
                        domains[q_id] = domain
            break # one file is enough for domains
    return domains

def calculate_metrics():
    configs = [
        # Gujarati precise splits
        {"name": "Gujarati", "in": "model_outputs_gujarati/precise/en", "results": f"{judge_dir}/guj/en", "lang": "en"},
        {"name": "Gujarati", "in": "model_outputs_gujarati/precise/gu", "results": f"{judge_dir}/guj/gu", "lang": "gu"},
        {"name": "Gujarati", "in": "model_outputs_gujarati/precise/hi", "results": f"{judge_dir}/guj/hi", "lang": "hi"},
        # Tamil precise splits
        {"name": "Tamil", "in": "model_outputs_tamil/precise_ta/en", "results": f"{judge_dir}/tam/en", "lang": "en"},
        {"name": "Tamil", "in": "model_outputs_tamil/precise_ta/ta", "results": f"{judge_dir}/tam/ta", "lang": "ta"},
        {"name": "Tamil", "in": "model_outputs_tamil/precise_ta/hi", "results": f"{judge_dir}/tam/hi", "lang": "hi"},
    ]

    for conf in configs:
        print(f"\n{'='*70}")
        print(f" Dataset: {conf['name']} | Language: {conf['lang'].upper()} ")
        print(f"{'='*70}")
        
        in_dir = conf["in"]
        res_dir = conf["results"]
        
        if not os.path.exists(in_dir):
            continue
            
        domains_map = load_domains(in_dir)
        
        if not os.path.exists(res_dir):
            print(f"(No results found in {res_dir})\n")
            continue
            
        json_results = [f for f in os.listdir(res_dir) if f.startswith("results_") and f.endswith(".json")]
        
        if not json_results:
            print(f"(No JSON evaluation results found in {res_dir})\n")
            continue
            
        for file in sorted(json_results):
            # Parse model name (e.g. results_gpt-5_gpt-4o-mini.json)
            base = file[len("results_"):-len(".json")]  # removes 'results_' and '.json'
            # Typically structure is {data_stem}_{judge_model}
            parts = base.split("_")
            
            # Identify where the judge model split happens, usually the last part unless judge model has underscores
            # e.g., 'gpt-5_gpt-4o-mini' -> 'gpt-5', 'gpt-4o-mini'
            if len(parts) >= 2:
                model_name = parts[0]
                judge_model = "_".join(parts[1:])
            else:
                model_name = base
                judge_model = "unknown"
                
            path = os.path.join(res_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                
            details = res_data.get("details", [])
            
            # Overall metics
            false_refusal = res_data.get("false_refusal", 0)
            hallu_rate = res_data.get("hallu_rate_not_abstain", 0)
            correct_rate = res_data.get("correct_rate", 0)
            
            print(f"\n[{model_name}]")
            print(f"Overall Metrics -> Acc: {correct_rate*100:.1f}% | Hallucination Rate: {hallu_rate*100:.1f}% | False Refusal: {false_refusal*100:.1f}%")
            
            domain_stats = {}
            for item in details:
                q_id = item["id"]
                domain = domains_map.get(q_id, "Unknown")
                if domain not in domain_stats:
                    domain_stats[domain] = {"total": 0, "correct": 0, "hallu": 0, "abstain": 0}
                
                domain_stats[domain]["total"] += 1
                if item["is_abstaining"]:
                    domain_stats[domain]["abstain"] += 1
                elif item["is_hallucinated"]:
                    domain_stats[domain]["hallu"] += 1
                else:
                    domain_stats[domain]["correct"] += 1
                    
            # Print per domain breakdown
            print(f"Per-Domain Breakdown:")
            for d in sorted(domain_stats.keys()):
                stats = domain_stats[d]
                d_total = stats["total"]
                d_acc = stats["correct"] / d_total if d_total > 0 else 0
                
                not_abstain = d_total - stats["abstain"]
                d_hallu = stats["hallu"] / not_abstain if not_abstain > 0 else 0
                d_abstain = stats["abstain"] / d_total if d_total > 0 else 0
                
                print(f"  - {d.ljust(35)}: Acc = {d_acc*100:4.1f}%  |  Hallu = {d_hallu*100:4.1f}%  |  False Refusal = {d_abstain*100:4.1f}%  (n={d_total})")

if __name__ == "__main__":
    calculate_metrics()
