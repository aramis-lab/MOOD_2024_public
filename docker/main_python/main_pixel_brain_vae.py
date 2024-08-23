from mood.data.create_caps import mood_to_clinicadl
from pathlib import Path
from clinicadl.predict.predict import predict
from clinicadl.extract.extract import DeepLearningPrepareData
from clinicadl.utils.clinica_utils import create_subs_sess_list
from clinicadl.extract.extract_utils import get_parameters_dict
import shutil
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--extract_json", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=False)
    parser.add_argument("--split", type=int)
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
            tmp_dir=args.output / "tmp",
        )


    maps_dir = args.maps
    shutil.copytree(maps_dir, args.output / "tmp" / "maps")
    maps_dir = args.output / "tmp" / "maps"
    print(f"The MAPS folder we'll be using is at: {maps_dir}")


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
    
    caps_output_path = maps_dir / f"split-{args.split}" / "best-loss" / "CapsOutput"
    create_subs_sess_list(caps_output_path, output_dir=caps_output_path, file_name="subjects_sessions.tsv", is_bids_dir=False)
    print(caps_output_path, Path(caps_output_path).is_dir())
    print("End of the prediction")

