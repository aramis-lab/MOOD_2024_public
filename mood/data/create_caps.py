import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


def mood_to_clinicadl(mood_path: Path, caps_path: Path, caps: bool = False, filter_subjects: Optional[Path] = None):
    if not isinstance(caps_path, Path):
        caps_path = Path(caps_path)

    if not isinstance(mood_path, Path):
        mood_path = Path(mood_path)

    if mood_path.is_dir():
        path_list = list(mood_path.glob("*.nii.gz"))
    else:
        raise FileNotFoundError(
            f"The path you give is not a directory: {mood_path}"
        )

    if caps_path.is_dir():
        raise FileExistsError(f"The path you give already exists: {caps_path}")
    else:
        (caps_path).mkdir(parents=True)

    if filter_subjects is not None:
        subject_filter = pd.read_csv(filter_subjects, sep='\t')['participant_id'].values

    columns = ["participant_id", "session_id", "diagnosis"]
    output_df = pd.DataFrame(columns=columns)
    session = "ses-M000"
    diagnosis = "CN"
    for mood_path in tqdm(path_list):

        subject = int((Path(mood_path.stem).stem).split("_")[-1])
        if subject < 10:
            subject_num = "00" + str(subject)
        elif 10 <= subject < 100:
            subject_num = "0" + str(subject)
        elif 100 <= subject:
            subject_num = str(subject)

        subject = "sub-" + subject_num
        if (filter_subjects is not None) and (subject not in subject_filter):
            continue

        if caps:
            subject_path = caps_path / "subjects" / subject / session / "custom"
            filename = f"sub-{subject_num}_ses-M000_mood.nii.gz"

        else:
            subject_path = caps_path / subject / session / "custom"
            filename = f"sub-{subject_num}_ses-M000_mood.nii.gz"

        row_df = pd.DataFrame([[subject, session, diagnosis]], columns=columns)
        output_df = pd.concat([output_df, row_df])

        subject_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=mood_path, dst=subject_path / filename)

    output_df.to_csv(caps_path / "subjects_sessions.tsv", sep="\t", index=False)

    return caps_path / "subjects_sessions.tsv"
