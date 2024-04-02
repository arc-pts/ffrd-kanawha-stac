import json
from pathlib import Path

def main():
    meta = Path('./meta')
    for path in meta.iterdir():
        if path.is_file():
            with open(path, 'r') as f:
                runtimes = []
                for line in f:
                    data = json.loads(line)
                    runtime = data["results_summary:computation_time_total_minutes"]
                    runtimes.append(runtime)
                print(f"{path.name} - average runtime: {sum(runtimes) / len(runtimes)}")

if __name__ == "__main__":
    main()
