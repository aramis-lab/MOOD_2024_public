


if __name__ == "__main__":

    import argparse
    from clinicadl.predict.predict import predict

    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", required=True, type=str)
    parser.add_argument("--tsv", required=True, type=str)
    parser.add_argument("--maps", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--dataset", type=bool, default=False)
    parser.add_argument("--splti", type=str)

    args = parser.parse_args()

    print(f"The given subjectq sessionq tsv is: {args.tsv}")
    print(f"The given caps path folder is: {args.caps}")
    print(f"The given maps path folder is: {args.maps}")
    print(f"The given dataset is: {args.dataset}")
    print(f"The given split is: {args.split}")

    print(f"Begining of the prediction on {args.caps}, with MAPS : {args.maps} (split-{args.split})")
    predict(maps_dir=args.maps,
            data_group=args.dataset,
            caps_directory=args.caps,
            tsv_path=args.tsv,
            gpu=True,
            split_list=[args.split],
            diagnoses=["AD", "CN"], # ???
            save_caps=True,
            skip_leak_check = True,
            overwrite = True,
        )
    print("End of the prediction")

