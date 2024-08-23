from mood.data.create_caps import mood_to_clinicadl
from pathlib import Path


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", required=True, type=str)
    parser.add_argument("--mood", required=True, type=str)

    args = parser.parse_args()

    print(f"The given input folder is: {args.mood}")
    print(f"The given caps folder is: {args.caps}")

    mood_to_clinicadl(args.mood, args.caps, caps = True)
    print(f"The CAPS path is: {args.caps}")
