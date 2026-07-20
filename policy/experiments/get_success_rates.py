import os
import csv

def extract_success_rate(log_path):
    """Extract the success rate from the last non-empty line of log.txt."""
    try:
        with open(log_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_line = lines[-1]
                if last_line.startswith("Success rate:"):
                    return float(last_line.split("Success rate:")[1].strip().rstrip('%'))
    except Exception as e:
        print(f"Failed to read {log_path}: {e}")
    return None

def collect_success_rates(base_dir):
    records = []

    for root, dirs, files in os.walk(base_dir):
        if 'log.txt' in files:
            rel_path = os.path.relpath(root, base_dir)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) != 5:
                print(f"Skipping {root}, expected 5 path components, got {len(path_parts)}")
                continue

            task, robot, model, checkpoint, action = path_parts
            model_dir = os.path.join(base_dir, task, robot, model)

            try:
                checkpoint_folders = [
                    d for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))
                ]
            except Exception as e:
                print(f"Skipping {model_dir}: {e}")
                continue

            if len(checkpoint_folders) != 3:
                print(f"Skipping {model_dir}, expected 3 checkpoints, got {len(checkpoint_folders)}")
                continue

            success_rate = extract_success_rate(os.path.join(root, 'log.txt'))
            if success_rate is not None:
                records.append(path_parts + [success_rate])

    return records


def write_csv(records, output_file):
    header = ['task', 'robot', 'model', 'checkpoint', 'action', 'success_rate']
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect success rates from log.txt files")
    parser.add_argument("--base_dir", default="rollouts/", help="Base directory to search")
    parser.add_argument("--output", default="success_rates.csv", help="Output CSV file")
    args = parser.parse_args()

    records = collect_success_rates(args.base_dir)
    write_csv(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")