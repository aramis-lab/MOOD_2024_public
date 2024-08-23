


if __name__ == "__main__":
    import argparse
    from clinicadl.extract.extract import DeepLearningPrepareData
    from clinicadl.extract.extract_utils import get_parameters_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", required=True, type=str)
    parser.add_argument("--tsv", required=True, type=str)
    parser.add_argument("--extract_json", type=str, required=True)
    parser.add_argument("--custom_suffix", type=str, required=True)
    parser.add_argument("--n_procs", type=int, required=False, default = 1)

    args = parser.parse_args()

    print(f"The given caps path folder is: {args.caps}")
    print(f"The given subjectq sessionq tsv is: {args.tsv}")
    print(f"The given extract json name is: {args.extract_json}")
    print(f"The given custom suffix name is: {args.custom_suffix}")
    print(f"The given number of procs is: {args.n_procs}")

    parameters = get_parameters_dict(
            "custom",
            "image",
            False,
            use_uncropped_image = False,
            extract_json = args.extract_json, #"extract_mood",
            custom_suffix = args.custom_suffix , #"*mood*",
        )
    DeepLearningPrepareData(
            caps_directory=args.caps,
            tsv_file=args.tsv,
            n_proc = args.n_procs,
            parameters=parameters,
        )
    print(f"pt tensors have been extracted successfully in caps directory : {args.caps_path}")
