from pathlib import Path 
from mood.scripts.mood_evalresults import eval_list
from mood.post_processing.process_residual_sample import ResidualType, get_residual
from mood.utils.exceptions import InvalidPathException
from mood.transforms.utils import save_image, extract_image
from typing import Union, List
import json 
import numpy as np
from clinicadl.utils.clinica_utils import clinicadl_file_reader, find_sub_ses_pattern_path, get_subject_session_list
import pandas as pd
import math
import shutil
#from clinicadl.transforms.utils import TransformsSampleLevel, TransformsPixelLevel



def compute_sample_gt(caps_test_path:Union[str, Path], dataset: str, subject_session_tsv: Union[str, Path], mood_input_path: Path):
    caps_test_path = Path(caps_test_path)
    subject_session_tsv = Path(subject_session_tsv)

    if not caps_test_path.is_dir():
        raise InvalidPathException(f"The Output Caps Path must be a directory: {caps_test_path}")
        
    
    if not subject_session_tsv.is_file():
        raise InvalidPathException(f"The subject session tsv file must exists: {subject_session_tsv}")

    output_pattern = "*custom/*mood.nii.gz"
    subject_session_df = pd.read_csv(subject_session_tsv, sep="\t")

    subjects_list = subject_session_df["participant_id"]
    sessions_list = subject_session_df["session_id"]
    sample_level_list = subject_session_df["image level"] 
    

    for (subject, session, sample_level) in zip(subjects_list, sessions_list, sample_level_list):
        list_ = []
        error = []
        find_sub_ses_pattern_path(input_directory=caps_test_path, subject=subject, session=session, pattern = output_pattern, results=list_, error_encountered=error, is_bids=False)
        path_subject = Path(list_[0])

        path_sample_gt = mood_input_path / f"{dataset}_label" / "sample"
        
        if not path_sample_gt.is_dir():
            path_sample_gt.mkdir(parents = True)

        number = (Path(path_subject.stem).stem)[0:16] 
        filename = path_sample_gt / f"{dataset}_{number}.nii.gz.txt"

        print(sample_level)
        if math.isnan(sample_level):
            print(number, "0")
            f = open(filename, "w")
            f.write("0")
            f.close()

        elif sample_level is True:
            print(number, "1")
            f = open(filename, "w")
            f.write("1")
            f.close()
        else: 
            print(number, "pb")


def compute_pixel_gt(caps_test_path:Union[str, Path], dataset: str, subject_session_tsv: Union[str, Path], mood_input_path: Path):  
    caps_test_path = Path(caps_test_path)
    subject_session_tsv = Path(subject_session_tsv)

    if not caps_test_path.is_dir():
        raise InvalidPathException(f"The Output Caps Path must be a directory: {caps_test_path}")
        
    
    if not subject_session_tsv.is_file():
        raise InvalidPathException(f"The subject session tsv file must exists: {subject_session_tsv}")

    output_pattern = "*custom/*mask.nii.gz"
    subject_session_df = pd.read_csv(subject_session_tsv, sep="\t")

    subjects_list = subject_session_df["participant_id"]
    sessions_list = subject_session_df["session_id"]
    sample_level_list = subject_session_df["image level"] 
    

    for (subject, session, sample_level) in zip(subjects_list, sessions_list, sample_level_list):
        list_ = []
        error = []
        find_sub_ses_pattern_path(input_directory=caps_test_path, subject=subject, session=session, pattern = output_pattern, results=list_, error_encountered=error, is_bids=False)
        path_subject = Path(list_[0])

        path_sample_gt = mood_input_path / f"{dataset}_label" / "pixel"
        
        if not path_sample_gt.is_dir():
            path_sample_gt.mkdir(parents = True)

        number = (Path(path_subject.stem).stem)[0:16] 
        filename = path_sample_gt / f"{dataset}_{number}.nii.gz"

        shutil.copyfile(path_subject, filename)


def compute_sample_pred(list_path_output: List[Path], mood_input_path: Path, mood_output_path: Path):

    # if caps_test_output_path.is_dir():

    #     subjects, sessions = get_subject_session_list(caps_test_output_path, subject_session_file = subjects_sessions_file, is_bids_dir = False)
    #     list_files, _ = clinicadl_file_reader(subjects, sessions, caps_test_output_path, information= {"pattern": pattern, "description": "pattern for mood 2024 challenge"})

    from os import listdir
    from os.path import isfile, join

    fichiers = [f for f in listdir(mood_input_path) if isfile(join(mood_input_path, f))]
    pattern = Path(fichiers[0]).name
    for path in list_path_output:
        path = Path(path)
        tmp_np = extract_image(path)
        
        tmp_np_norm = (tmp_np - tmp_np.min()) / (tmp_np.max() - tmp_np.min()) #??

        out = np.mean(np.abs(tmp_np_norm))

        out_path = mood_output_path / (pattern.split("_")[0] + f"_{path.name[4:7]}." + pattern.split(".",1)[1] +".txt")
        f = open(out_path, "w")
        f.write(str(out))
        f.close()

        print(out)






def compute_pixel_pred(list_path_output: List[Path], mood_input_path: Path, mood_output_path: Path):

    from os import listdir
    from os.path import isfile, join

    fichiers = [f for f in listdir(mood_input_path) if isfile(join(mood_input_path, f))]
    pattern = Path(fichiers[0]).name
    for path in list_path_output:
        path = Path(path)
        out_path = mood_output_path / (pattern.split("_")[0] + f"_{path.name[3:6]}." + pattern.split(".",1)[1])

        shutil.copyfile(path, out_path)


