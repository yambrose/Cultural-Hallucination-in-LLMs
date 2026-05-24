import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="Model to use for judging (default is gpt-4.1)")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel runs. Keep low to avoid rate limits.")
    args = parser.parse_args()

    specific_files = [
        # {"data_path": "model_outputs_tamil/precise_ta/ta/sarvam-105b.jsonl", "out": "judge_results_new/tam/ta"},
        {"data_path": "model_outputs_gujarati/precise/en/llama-4-maverick.jsonl", "out": "judge_results_new/guj/en"},
    ]
    
    # # Also evaluate ALL the newly-restored precise_ta/hi files EXCEPT llama/qwen
    hi_dir = "model_outputs_tamil/precise_ta/hi"
    ignore_files = ["qwen-mt-plus.jsonl", "llama-4-maverick.jsonl"]
    for file in os.listdir(hi_dir):
        if file.endswith(".jsonl") and file not in ignore_files:
            specific_files.append({
                "data_path": os.path.join(hi_dir, file),
                "out": "judge_results_new/tam/hi"
            })

    commands = []
    for conf in specific_files:
        data_path = conf["data_path"]
        out_dir = conf["out"]
        if not os.path.exists(data_path):
            print(f"Info: {data_path} does not exist, skipping.")
            continue
            
        cmd = [
            "python3", "eval_precisewiki.py",
            "--data_path", data_path,
            "--save_dir", out_dir,
            "--judge_model", args.judge_model
        ]
        commands.append(cmd)

    print(f"Total evaluation files to process: {len(commands)}")
    print(f"Running with max_workers={args.max_workers}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(run_cmd, commands)

if __name__ == "__main__":
    main()
