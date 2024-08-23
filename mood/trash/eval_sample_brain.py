from mood.data.create_caps import mood_to_clinicadl
import sys
from pathlib import Path
import pandas as pd
from mood.post_processing.process_residual_sample import ResidualType
from mood.post_processing.save_sample_score import compute_sample_pred, compute_sample_gt, compute_pixel_gt, compute_pixel_pred, create_residual
from clinicadl.predict.predict import predict
from clinicadl.extract.extract import DeepLearningPrepareData
from clinicadl.extract.extract_utils import get_parameters_dict
import shutil
from mood.scripts.mood_evalresults import eval_dir


path_sample_pred = Path(sys.argv[1]) #/ "toy"
print(f"The given input folder is (prediction): {path_sample_pred}")

path_sample_label = Path(sys.argv[2])
print(f"The given output folder is (label): {path_sample_label}")


score = eval_dir(pred_dir= Path(path_sample_pred), label_dir=Path(path_sample_label), mode="sample", save_file=None)
print(f"AP sample level is {score}")