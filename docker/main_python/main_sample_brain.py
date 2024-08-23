from mood.data.create_caps import mood_to_clinicadl
from pathlib import Path
from mood.scripts.create_residual import create_residual
from mood.post_processing.save_sample_score import compute_sample_pred
from clinicadl.predict.predict import predict
from clinicadl.extract.extract import DeepLearningPrepareData
from clinicadl.extract.extract_utils import get_parameters_dict
import shutil
import argparse

if __name__ == "__main__":

    import datetime
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--extract_json", type=str, required=True)
    parser.add_argument("--custom_suffix", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=False)
    parser.add_argument("--split", type=int)
    parser.add_argument("--mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--mean_path", type=Path)
    parser.add_argument("--std_path", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    print(f"The given input folder is: {args.input}")
    print(f"The given output folder is: {args.output}")
    
    caps_path = args.output / "tmp" / "caps_brain"
    mood_to_clinicadl(args.input, caps_path, caps = True)
    print(f"The CAPS path is: {caps_path}")

    subjects_sessions_tsv = caps_path / "subjects_sessions.tsv" #mood_to_clinicadl(args.input, caps_path, caps = True)
    print(f"The tsv file corresponding to the list of participants for testing is at: {subjects_sessions_tsv}")

    import multiprocessing
    procs = multiprocessing.cpu_count()
    print(f"the number of procs is: {procs}")

    parameters = get_parameters_dict(
            "custom",
            "image",
            False,
            use_uncropped_image = False,
            extract_json = args.extract_json, #"extract_mood",
            custom_suffix = "*mood*", #args.custom_suffix, #"*mood*",
        )
    DeepLearningPrepareData(
            caps_directory=caps_path,
            tsv_file=subjects_sessions_tsv,
            n_proc = procs,
            parameters=parameters,
            tmp_dir=args.output / "tmp"
        )


    maps_dir = args.maps
    shutil.copytree(maps_dir, args.output / "tmp" / "maps")
    maps_dir = args.output / "tmp" / "maps"
    print(f"The MAPS folder we'll be using is at: {maps_dir}")

    import torch
    print(f"Begining of the prediction on {caps_path}, with MAPS : {maps_dir} (split-{args.split})")
    predict(maps_dir=maps_dir,
            data_group=args.dataset,
            caps_directory=caps_path,
            tsv_path=subjects_sessions_tsv,
            gpu=torch.cuda.is_available(),
            split_list=[args.split],
            diagnoses=["AD", "CN"], # ???
            save_caps=True,
            skip_leak_check = True,
            overwrite = True,
        )
    print("End of the prediction")

    caps_output = maps_dir / f"split-{args.split}" / "best-loss" / "CapsOutput"
    print(f"The CAPS output folder is created at: {caps_output}")

    list_output_residual_post_processed = create_residual(caps_test_output_path = caps_output, subjects_sessions_file = subjects_sessions_tsv,mode = "sample", mean_path = args.mean_path, std_path = args.std_path)
    print(f"The residual images have been computed")

    path_sample_pred = compute_sample_pred(list_output_residual_post_processed, mood_input_path=args.input, mood_output_path = args.output)
    print(f"Sample predictions have been computed at {path_sample_pred}")

    shutil.rmtree(args.output / "tmp", ignore_errors=True)

    print(datetime.datetime.now())