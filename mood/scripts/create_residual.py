
import sys
import numpy as np
from typing import Optional
from pathlib import Path
from clinicadl.utils.clinica_utils import clinicadl_file_reader, find_sub_ses_pattern_path, get_subject_session_list
from mood.transforms.utils import save_image, extract_image
from mood.post_processing.process_residual_pixel import post_processing as post_processing_pixel
from mood.post_processing.process_residual_sample import post_processing as post_processing_sample



def create_residual(caps_test_output_path: Path, subjects_sessions_file: Path, mode:str, mean_path:Optional[Path], std_path:Optional[Path], brain:bool=True):

    """
    Input and Output files MUST be in the CAPS directory provided.

    """

    if caps_test_output_path.is_dir():

        subjects, sessions = get_subject_session_list(caps_test_output_path, subject_session_file = subjects_sessions_file, is_bids_dir = False)
        list_files, _ = clinicadl_file_reader(subjects, sessions, caps_test_output_path, information= {"pattern": "*custom/*_input.nii.gz", "description": "pattern for mood 2024 challenge"})

        list_output_files = []

        for input_path in list_files:
            output_path = str(input_path).replace("input", "output")
            output_path = Path(output_path)
            output_np = extract_image(output_path)

            input_path = Path(input_path)
            input_np = extract_image(input_path)

            if mode == "sample":
                mean_np = extract_image(mean_path)
                std_np = extract_image(std_path)
                residual_np = post_processing_sample(input_np, output_np, mean_np, std_np)

            elif mode == "pixel":
                thresh =  3.67
                residual_np = post_processing_pixel(input_np, output_np, mean_np, std_np, thresh, brain=brain)
            
            save_image(residual_np, Path(str(input_path).replace("input", "residual")))

            list_output_files.append(Path(str(input_path).replace("input", "residual")))

    else :
        list_output_files = None

    return list_output_files

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", required=True, type=str)
    parser.add_argument("--tsv", required=True, type=str)
    parser.add_argument("--mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--mean_path", type=str)
    parser.add_argument("--std_path", type=str)

    args = parser.parse_args()

    print(f"The given input folder is: {args.caps_output}")
    print(f"The given tsv file is: {args.tsv}")
    print(f"TWorking for {args.mode} task")

    create_residual(caps_test_output_path=Path(args.caps_output), subjects_sessions_file=Path(args.subjects_sessions_tsv), mode = args.mode, mean_path=Path(args.mean_path), std_path = Path(args.std_path))
    print("The residual images have been computed")

