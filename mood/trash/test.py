from pathlib import Path 
# from mood_evalresults import eval_list
from typing import Union
import json 
from clinicadl.utils.clinica_utils import clinicadl_file_reader, find_sub_ses_pattern_path
import pandas as pd
from mood.transforms.utils import TransformsSampleLevel, TransformsPixelLevel



class InvalidPathException(Exception):
    pass

class InvalidArgumentException(Exception):
    pass

class InvalidTransformException(Exception):
    pass

def get_eval_score(caps_input_path: Union[str, Path], caps_output_path: Union[str, Path], subject_session_tsv: Union[str, Path], postprocessing_json: Union[str, Path], mode:str= "pixel"):
    caps_input_path = Path(caps_input_path)
    caps_output_path = Path(caps_output_path)
    subject_session_tsv = Path(subject_session_tsv)

    if not caps_input_path.is_dir() :
        raise InvalidPathException(f"The Input Caps Path must be a directory: {caps_input_path}")
    
    if not caps_output_path.is_dir():
        raise InvalidPathException(f"The Output Caps Path must be a directory: {caps_output_path}")
        
    if not subject_session_tsv.is_file():
        raise InvalidPathException(f"The subject session tsv file must exists: {subject_session_tsv}")
    

    #preprocessing_dict = get_processing_dict(caps_input_path, preprocessing_json)
    postprocessing_dict =get_processing_dict(caps_output_path, postprocessing_json)
    output_pattern = postprocessing_dict["file_type"]["pattern"]

    subject_session_df = pd.read_csv(subject_session_tsv, sep="\t")
    #subject_session_df.set_index(["participant_id", "session_id"], inplace=True)

    subjects_list = subject_session_df["participant_id"]
    #sessions_list = subject_session_df["session_id"]  
    transforms_list = subject_session_df["type"] 

    path_mask_list = []
    path_map_list = []
    error_encountered =[]
    for (subject, type) in zip(subjects_list, transforms_list):
        input_pattern = type + "_mask"

        if type in [e.value for e in TransformsPixelLevel]:
            find_sub_ses_pattern_path(input_directory=caps_input_path, subject=subject, session="ses-M000", pattern = "*" +input_pattern + ".nii.gz", results= path_mask_list, error_encountered=error_encountered, is_bids=False)
        elif type in [e.value for e in TransformsSampleLevel]:
            path_mask_list.append(caps_input_path / "mask" / (input_pattern + ".nii.gz"))
        else: 
            raise InvalidTransformException(f"We can't find the maskfor the transform: {type}")
        
        find_sub_ses_pattern_path(input_directory=caps_output_path, subject=subject, session="ses-M000", pattern = output_pattern, results= path_map_list, error_encountered=error_encountered, is_bids=False)

    return eval_list(pred_file_list=path_map_list,label_file_list=path_mask_list, mode = mode)


def get_processing_dict(caps_directory: Path, processing_json: Union[str, Path] = "")->dict:

    if processing_json== "": 
        raise InvalidArgumentException("You must give a postprocessing.json")
    
    processing_json = Path(processing_json)

    if processing_json.stem != "json":
        processing_json = Path(str(processing_json)+ ".json")

    processing_json = caps_directory / "tensor_extraction" / processing_json

    with processing_json.open(mode="r") as f:
        postprocessing_dict = json.load(f)
    
    return postprocessing_dict

    

if __name__ == "__main__":

    

    # ### PIXEL

    caps_inpout_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/caps_brain")
    caps_outpout_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/caps_brain_output")
    tsv_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/tsv/subjects_sessions.tsv")

    print(get_eval_score(caps_input_path=caps_inpout_path, caps_output_path=caps_outpout_path, subject_session_tsv=tsv_path, postprocessing_json="extract_image"))


    pred_label_pixel_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/brain/brain_label/pixel")
    toy_label_pixel_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/brain/toy_label/pixel")


    if pred_label_pixel_path.is_dir():
        pred_label_pixel_list = list(pred_label_pixel_path.glob("*.nii.gz"))

    if toy_label_pixel_path.is_dir():
        toy_label_pixel_list = list(toy_label_pixel_path.glob("*.nii.gz"))

    print(eval_list(pred_file_list=pred_label_pixel_list,label_file_list=toy_label_pixel_list))


    # ### SAMPLE

    pred_label_sample_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/brain/brain_label/sample")
    toy_label_sample_path = Path("/Users/camille.brianceau/aramis/MOOD24/DATA/brain/toy_label/sample")

    if pred_label_sample_path.is_dir():
        pred_label_sample_list = list(pred_label_sample_path.glob("*.nii.gz.txt"))

    if toy_label_sample_path.is_dir():
        toy_label_sample_list = list(toy_label_sample_path.glob("*.nii.gz.txt"))

    print(eval_list(pred_file_list=pred_label_sample_list,label_file_list=toy_label_sample_list, mode = "sample"))
