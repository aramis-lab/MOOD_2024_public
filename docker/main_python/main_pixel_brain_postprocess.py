
import os
from pathlib import Path
from mood.scripts.create_residual import create_residual
from mood.post_processing.save_sample_score import compute_pixel_pred


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--caps_output", required=True, type=Path)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--mean_path", type=Path)
    parser.add_argument("--std_path", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    list_output_residual_post_processed = create_residual(caps_test_output_path = args.caps_output, subjects_sessions_file = args.tsv, mode = "pixel", mean_path = args.mean_path, std_path = args.std_path)
    print("The residual images have been computed")

    path_pixel_pred = compute_pixel_pred(list_output_residual_post_processed, mood_input_path = args.input, mood_output_path = args.output)
    print(f"Sample predictions have been computed at {path_pixel_pred}")

    os.rmdir(args.output / "tmp")