import argparse

from decorrelated_ensemble.evaluation.runner import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_experiment(args.config, run_id=args.run_id))
