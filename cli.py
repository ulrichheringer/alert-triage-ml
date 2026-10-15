"""CLI for the reproducible classical ML pipeline."""
import argparse
import json
from pathlib import Path

import joblib

from pipeline import evaluate, predict, sample_input, train_model


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and run the classical ML MVP.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate", help="evaluate on the documented holdout")
    train = commands.add_parser("train", help="fit and save a model")
    train.add_argument("--model", default="model.joblib")
    inference = commands.add_parser("predict", help="predict from JSON input")
    inference.add_argument("--model", default="model.joblib")
    inference.add_argument("--input", help="JSON value; uses a synthetic example if omitted")
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate()
    elif args.command == "train":
        joblib.dump(train_model(), args.model)
        result = {"saved_model": str(Path(args.model))}
    else:
        model = joblib.load(args.model)
        result = predict(model, json.loads(args.input) if args.input else sample_input())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
