import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from clinicadl.utils.maps_manager import MapsManager
from mood.metrics.pixel_level import PixelF1Score, ThresholdFinder
from mood.post_processing.process_residual_pixel import PostProcessor
from mood.scripts.utils import ImageSaver, get_dataloader, mean_and_confidence


def compute_f1_score(
    maps_dir: Path,
    preprocessing: Path,
    data_group: str,
    data_group_val: str,
    postprocessing: Callable[[np.ndarray], np.ndarray],
    n_proc: int,
    batch_size: int,
    split_list: list[int] = None,
    threshold: Optional[float] = None,
    n_points: int = 10,
    threshold_range: tuple = (0.1, 0.9),
    min_size: int = 600,
    output_name: str = 'output',
    selection_metrics: str = None,
    save_preds: bool = False,
):
    maps_manager = MapsManager(maps_dir)

    if not split_list:
        split_list = maps_manager._find_splits()

    for split in split_list:
        if not selection_metrics:
            split_selection_metrics = maps_manager._find_selection_metrics(split)[0]
        else:
            split_selection_metrics = selection_metrics[0]

        caps_pred_dir = Path(
            maps_dir,
            f"split-{split}",
            f"best-{split_selection_metrics}",
            "CapsOutput",
        )
        dataloader = get_dataloader(
            maps_manager,
            output_name,
            preprocessing,
            str(Path(data_group) / f"split-{split}"),
            split,
            1,
            batch_size,
            split_selection_metrics,
        )
        val_dataloader = get_dataloader(
            maps_manager,
            output_name,
            preprocessing,
            str(Path(data_group_val) / f"split-{split}"),
            split,
            1,
            batch_size,
            split_selection_metrics,
        )

        threshold_finder = ThresholdFinder(process_fct=postprocessing, n_proc=n_proc, n_points=n_points, threshold_range=threshold_range, min_size=min_size)
        if threshold is None:
            threshold_finder.reset()
            for batch in val_dataloader:
                masks = batch["gt"]["data"].squeeze(1).numpy().astype(np.int8)
                inputs = batch["input"]["data"].squeeze(1).numpy()
                outputs = batch["output"]["data"].squeeze((1,2)).numpy()
                threshold_finder.append(inputs, outputs, masks, infos={
                        "participant_id": batch["input"]["participant_id"],
                        "session_id": batch["input"]["session_id"],
                    })

            threshold, scores = threshold_finder.aggregate()

            json_path = Path(
                maps_dir,
                f"split-{split}",
                f"best-{split_selection_metrics}",
                data_group_val,
                f"thresholds_{output_name}.json",
            )
            json_path.parent.mkdir(exist_ok=True)
            with open(json_path, "w+") as f:
                json.dump(scores, f, indent=4)
            print(f"best threshold: {threshold}, details saved in {json_path}")

        all_times = []
        img_saver = ImageSaver(prefix=caps_pred_dir, suffix="pred", n_proc=n_proc)
        metric = PixelF1Score(n_proc=n_proc, min_size=min_size)
        postprocessor = PostProcessor(process_fct=postprocessing, n_proc=n_proc, threshold=threshold)
        for batch in dataloader:
            masks = batch["gt"]["data"].squeeze(1).numpy().astype(np.int8)
            inputs = batch["input"]["data"].squeeze(1).numpy()
            outputs = batch["output"]["data"].squeeze((1,2)).numpy()

            preds, times = postprocessor.process(inputs, outputs)
            metric.append(preds, masks, infos={
                    "participant_id": batch["input"]["participant_id"],
                    "session_id": batch["input"]["session_id"],
                })
            all_times += times

            if save_preds:
                pred_batch = {
                    "data": preds,
                    "participant_id": batch["input"]["participant_id"],
                    "session_id": batch["input"]["session_id"],
                }
                img_saver.save_batch(pred_batch)

        matrix, results_per_img = metric.aggregate()
        mean_time, conf_time = mean_and_confidence(all_times)

        root = Path(
            maps_dir,
            f"split-{split}",
            f"best-{split_selection_metrics}",
            data_group,
        )
        root.mkdir(exist_ok=True)
        results_per_img.to_csv(root / f"detailed_results_{output_name}.tsv", sep="\t", index=False)
        df = pd.DataFrame(
            index=["Detected", "Not Detected", "F1-score", "threshold"],
            columns=["True", "False"],
        )
        df.index.name = "Anomaly"
        df.loc["Detected", "True"] = matrix["TP"]
        df.loc["Detected", "False"] = matrix["FP"]
        df.loc["Not Detected", "True"] = matrix["FN"]
        df.loc["F1-score", "True"] = matrix["F1-score"]
        df.loc["threshold", "True"] = threshold
        df.loc["post-processing time", "True"] = f"{mean_time} ± {conf_time}"
        df = df.reset_index()
        df.to_csv(root / f"confusion_matrix_{output_name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    from mood.post_processing.process_residual_pixel import post_processing

    maps_dir = Path("/root_dir/maps/GANs/pix2pix/attn_unet_sobel_aug_100")
    preprocessing = Path(
        "/root_dir/misc/extract_residual.json"
    )
    compute_f1_score(
        maps_dir,
        preprocessing,
        data_group="test_pixel",
        data_group_val="validation_post_processing",
        postprocessing=post_processing,
        n_proc=16,
        batch_size=16,
        split_list=[4],
        threshold=3.0,
        n_points=10,
        threshold_range=(1.0, 5.0),
        min_size=600,
        output_name='MS_BetaVAE_output_1',
        save_preds=False,
    )
