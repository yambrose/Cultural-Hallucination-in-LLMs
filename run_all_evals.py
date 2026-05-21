import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="gpt-4.1", help="Model to use for judging (default is gpt-4o-mini to limit costs/delays)")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel runs. Keep low to avoid rate limits.")
    args = parser.parse_args()

    configs = [
        # Gujarati precise splits
        {"in": "model_outputs_gujarati/precise/en", "out": "judge_results_new/guj/en"},
        {"in": "model_outputs_gujarati/precise/gu", "out": "judge_results_new/guj/gu"},
        {"in": "model_outputs_gujarati/precise/hi", "out": "judge_results_new/guj/hi"},
        # Tamil precise splits
        {"in": "model_outputs_tamil/precise_ta/en", "out": "judge_results_new/tam/en"},
        {"in": "model_outputs_tamil/precise_ta/ta", "out": "judge_results_new/tam/ta"},
        {"in": "model_outputs_tamil/precise_ta/hi", "out": "judge_results_new/tam/hi"},
    ]

    commands = []
    for conf in configs:
        in_dir = conf["in"]
        out_dir = conf["out"]
        if not os.path.exists(in_dir):
            print(f"Info: {in_dir} does not exist, skipping.")
            continue
            
        for file in os.listdir(in_dir):
            if file.endswith(".jsonl"):
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
