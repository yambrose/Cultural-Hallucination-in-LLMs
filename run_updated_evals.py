import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="Model to use for judging (default is gpt-4o-mini to limit costs/delays)")
    parser.add_argument("--max_workers", type=int, default=3, help="Number of parallel runs. Keep low to avoid rate limits.")
    args = parser.parse_args()

    configs = [
        # Only Tamil precise splits for the models we updated
        {"in": "model_outputs_tamil/precise_ta/en", "out": "judge_results_new/tam/en"},
        {"in": "model_outputs_tamil/precise_ta/ta", "out": "judge_results_new/tam/ta"},
        {"in": "model_outputs_tamil/precise_ta/hi", "out": "judge_results_new/tam/hi"},
    ]
    
    ignore_files = ["qwen-mt-plus.jsonl", "llama-4-maverick.jsonl"]

    commands = []
    for conf in configs:
        in_dir = conf["in"]
        out_dir = conf["out"]
        if not os.path.exists(in_dir):
            print(f"Info: {in_dir} does not exist, skipping.")
            continue
            
        for file in os.listdir(in_dir):
            if file.endswith(".jsonl") and file not in ignore_files:
                data_path = os.path.join(in_dir, file)
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
