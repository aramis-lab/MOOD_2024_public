from mood.data.create_caps import mood_to_clinicadl
import sys
from pathlib import Path
import pandas as pd
from mood.post_processing.process_residual_sample import ResidualType
from mood.post_processing.save_sample_score import compute_sample_pred, compute_sample_gt, compute_pixel_gt, compute_pixel_pred
from clinicadl.predict.predict import predict
from clinicadl.extract.extract import DeepLearningPrepareData
from clinicadl.extract.extract_utils import get_parameters_dict
import shutil
from mood.scripts.mood_evalresults import eval_dir


mood_input_path = Path(sys.argv[1]) #/ "toy"
print(f"The given input folder is: {mood_input_path}")

mood_output_path = Path(sys.argv[2])
print(f"The given output folder is: {mood_output_path}")

subjects_sessions_tsv = caps_path = mood_output_path / "tmp" / "caps_brain"
mood_to_clinicadl(mood_input_path, caps_path, caps = True)
print(f"The CAPS path is: {caps_path}")

subjects_sessions_tsv = caps_path / "subjects_sessions.tsv" #mood_to_clinicadl(mood_input_path, caps_path, caps = True)
print(f"The tsv file corresponding to the list of participants for testing is at: {subjects_sessions_tsv}")

parameters = get_parameters_dict(
        "custom",
        "image",
        False,
        use_uncropped_image = False,
        extract_json = "extract_mood",
        custom_suffix ="*mood*",
    )
DeepLearningPrepareData(
        caps_directory=caps_path,
        tsv_file=subjects_sessions_tsv,
        n_proc = 1,
        parameters=parameters,
    )

maps_dir = Path("/Users/camille.brianceau/aramis/MOOD24/MOOD_2024/maps")
import shutil
shutil.copytree(maps_dir, mood_output_path / "tmp" / "maps")
maps_dir = mood_output_path / "tmp" / "maps"
print(f"The MAPS folder we'll be using is at: {maps_dir}")

### PARAMETERTS ###

residual_type = ResidualType.VAL
dataset = "toy"
split = 4
shapes = (256, 256, 256)  #(169, 208, 179) # for VAE classic and mood data (256, 256, 256) # (169, 208, 179) for t1-linear and (128, 128, 128) for GAN

##################

print(f"Begining of the prediction on {caps_path}, with MAPS : {maps_dir} (split-{split})")
predict(maps_dir=maps_dir,
        data_group=dataset,
        caps_directory=caps_path,
        tsv_path=subjects_sessions_tsv,
        gpu=True,
        split_list=[split],
        diagnoses=["AD", "CN"], # ???
        save_caps=True,
        skip_leak_check = True,
        overwrite = True,
        batch_size = 1,
    )
print("End of the prediction")

caps_output = maps_dir / f"split-{split}" / "best-loss" / "CapsOutput"
print(f"The CAPS output folder is created at: {caps_output}")

create_residual(caps_test_output_path = caps_output, subjects_sessions_file = subjects_sessions_tsv)
print("The residual (output-input) have been computed")

path_sample_pred = compute_sample_pred(caps_test_output_path = caps_output, subjects_sessions_file = subjects_sessions_tsv, residual_type = residual_type, dataset = dataset, mood_output_path = mood_output_path)
print(f"Sample predictions have been computed at {path_sample_pred}")
