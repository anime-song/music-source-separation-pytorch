import pickle
import argparse
import pickle
import os
import csv
import wave
from pathlib import Path


def create_dataset(dataset_path, dataset_meta_csv_path):
    dataset_meta_csv_path = Path(dataset_meta_csv_path)

    samples_all = []

    with dataset_meta_csv_path.open(encoding="utf-8") as f:
        # metaInfo = json.load(f)
        meta_info = csv.DictReader(f)
        for e in meta_info:
            print(e)
            audioPath = os.path.join(dataset_path, e["audio_filename"])

            with wave.open(audioPath) as f:
                fs = f.getframerate()
                nSamples = f.getnframes()
                nChannel = f.getnchannels()

            e["fs"] = fs
            e["nSamples"] = nSamples
            e["nChannel"] = nChannel
            samples_all.append(e)

    return samples_all


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("dataset_path", help="folder path to the maestro dataset")
    argumentParser.add_argument("metadata_csv_path", help="path to the metadata file of the maestro dataset (csv)")
    argumentParser.add_argument("output_path", help="path to the output folder")

    args = argumentParser.parse_args()

    dataset_path = args.dataset_path
    dataset_meta_csv_path = args.metadata_csv_path
    output_path = args.output_path

    dataset = create_dataset(dataset_path, dataset_meta_csv_path)

    train = []
    val = []
    test = []
    for e in dataset:
        if e["split"] == "train":
            train.append(e)
        elif e["split"] == "validation":
            val.append(e)
        elif e["split"] == "test":
            test.append(e)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, "train.pickle"), "wb") as f:
        for sample in train:
            pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, "val.pickle"), "wb") as f:
        for sample in val:
            pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, "test.pickle"), "wb") as f:
        for sample in test:
            pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)
