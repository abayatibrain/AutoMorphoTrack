"""Command line entry point with a retained JSON configuration."""

import argparse
import json
from .config import PipelineConfig
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Analyze an explicitly configured two-channel image sequence")
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True, help="JSON PipelineConfig parameters")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = PipelineConfig.from_dict(json.load(handle))
    result = run_pipeline(args.input, args.out, config)
    print(json.dumps({"run_id": result["manifest"]["run_id"], "out": args.out}))


if __name__ == "__main__":
    main()
