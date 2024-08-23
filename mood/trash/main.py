from pathlib import Path
import pandas as pd
from mood.post_processing.process_residual_sample import create_residual, ResidualType
from mood.post_processing.save_sample_score import compute_sample_pred, compute_sample_gt, compute_pixel_gt, compute_pixel_pred

#from mood.post_processing.process_residual_pixel import median_filter
from mood.scripts.mood_evalresults import eval_dir
from mood.post_processing.process_residual_pixel import PostProcessor

if __name__ == "__main__":

    # General path 
    caps_test_input_path = Path("/root_dir/data/brain/caps_test")
    subject_session_tsv = Path("/root_dir/data/brain/caps_test/sample_level.tsv")
    subject_session_pixel_tsv = Path("/root_dir/data/brain/caps_test/pixel_level.tsv")


    # caps_test_input_path = Path("/root_dir/data/brain/caps_test")
    # subject_session_tsv = Path("/root_dir/data/brain/caps_test/sample_level.tsv")
    # subject_session_pixel_tsv = Path("/root_dir/data/brain/caps_test/pixel_level.tsv")

    mood_input_path = Path("/root_dir/data/mood_inputs/brain")

    ### ARG TO CHOSE ###

    #maps_path = Path("/root_dir/maps/maps_test/")
    maps_path = Path("/root_dir/maps/MAPS_MS_BetaVAE_0")
    #maps_path = Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_25")
    #maps_path = Path("/root_dir/maps/MAPS_MS_BetaVAE_T1-linear_QC_0")
    #maps_path = Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100")
    residual_type = ResidualType.VAL
    dataset = "test"
    split = 2
    shapes = (256, 256, 256) #(169, 208, 179) # for VAE classic and mood data (256, 256, 256) # (169, 208, 179) for t1-linear and (128, 128, 128) for GAN

    #### END ARG TO CHOSE


    # Uncomment if you want to regenerate sample gt for another caps_test for example 
    # compute_sample_gt(caps_test_input_path, dataset, subject_session_tsv, mood_input_path)
    # print("compute sample ok")
    
    # compute_pixel_gt(caps_test_input_path, dataset, subject_session_pixel_tsv, mood_input_path)
    # print("compute pixel ok")


    mood_output_sample_path = maps_path / f"split-{split}" / "best-loss" / dataset
    caps_test_output_path = maps_path / f"split-{split}" / "best-loss" / "CapsOutput"
    

    output_df = pd.DataFrame(columns = ["level", "residual_type", "pred"])
    output_df.set_index(["level", "residual_type"], inplace = True)
        
    # if the residual - mean val / std val 
    if residual_type != ResidualType.CLASSIC:
        if not (caps_test_output_path.parent / "residual").is_dir() or not any((caps_test_output_path.parent / "residual").iterdir()):
            create_residual(caps_input_path = caps_test_input_path, residual_type = residual_type, maps_path = maps_path, split = split, shapes = shapes)#, shapes = (256, 256, 256)) # (169, 208, 179)
            print(f"create reisudal for {residual_type.value}")

    # to put the prediction in a txt file in the mood output directory and the nii gz file in the pixel_level folder
    compute_sample_pred(caps_test_output_path = caps_test_output_path, subjects_sessions_file = subject_session_tsv, residual_type = residual_type, dataset = dataset, mood_output_path = mood_output_sample_path)
    print(f"compute sample pred for {residual_type.value}")

    # to put the prediction in a txt file in the mood output directory and the nii gz file in the pixel_level folder
    compute_pixel_pred(caps_test_output_path = caps_test_output_path, subjects_sessions_file = subject_session_pixel_tsv,residual_type = residual_type, dataset = dataset, mood_output_path = mood_output_sample_path)
    print(f"compute pixel pred for {residual_type.value}")

    # calcul score with and without median filter (can be change if we want others or more postprocessing)
    postprocessor = PostProcessor(process_fct=median_filter) # ????

    # calcul AP for sample level
    score = eval_dir(label_dir = mood_input_path / f"{dataset}_label"/ "sample", pred_dir = mood_output_sample_path / f"{dataset}_label"/ f"{residual_type.value}" / "sample", mode = "sample")
    print(residual_type.value, score)
    output_df.loc[("sample", residual_type.value), "pred"] = str(score)


    score_pixel = eval_dir(label_dir = mood_input_path / f"{dataset}_label"/ "pixel", pred_dir = mood_output_sample_path / f"{dataset}_label"/ f"{residual_type.value}" / "pixel", mode = "pixel")
    print(residual_type.value, score_pixel)
    output_df.loc[("pixel", residual_type.value), "pred"] = str(score_pixel)

    tsv_path = Path(
        maps_path,
        f"split-{split}",
        "best-loss", 
        "test",
        f"AP_{residual_type.value}_results.tsv",
    )
    tsv_path.parent.mkdir(exist_ok=True)
    output_df.to_csv(tsv_path, sep="\t", index=False)


    