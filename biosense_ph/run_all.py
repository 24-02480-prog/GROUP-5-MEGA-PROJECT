"""Run the complete BioSense PH prototype pipeline."""

from __future__ import annotations

import argparse

from src.t08_integration import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioSense PH demo pipeline")
    parser.add_argument(
        "--data-source",
        choices=("auto", "real", "synthetic"),
        default="auto",
        help="Select real data, synthetic stand-ins, or automatic fallback",
    )
    arguments = parser.parse_args()
    run_pipeline(data_source=arguments.data_source)


if __name__ == "__main__":
    main()
