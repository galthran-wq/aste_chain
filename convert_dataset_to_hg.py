import argparse
from typing import List, Tuple
from ast import literal_eval
from pathlib import Path

from datasets import Dataset, DatasetDict

from data_utils import get_extraction_aste_targets


def parse_example(line: str):
    text, annots = line.split("####")
    triplets: List[Tuple[List[int], List[int], str]] = literal_eval(annots.strip())
    tokenized_text = text.split()
    return tokenized_text, triplets


def example_to_string(example_text, triplets):
    aspects =[
        " ".join(example_text[token_i] for token_i in triplet[0])
        for triplet in triplets
    ] 
    opinions = [
        " ".join(example_text[token_i] for token_i in triplet[1])
        for triplet in triplets
    ]
    polarities = [
        triplet[2]
        for triplet in triplets
    ]
    return " ".join(example_text), [
        tuple(triplet) for triplet in list(zip(aspects, opinions, polarities))
    ]


def parse_example_to_string(line):
    example_text, example_triplets = parse_example(line)
    return example_to_string(example_text, example_triplets)


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    testlines = open(src / "test.txt").readlines()
    trainlines = open(src / "train.txt").readlines()
    vallines = open(src / "dev.txt").readlines()
    dataset_dict = {}
    for name, lines in zip(["train", "val", "test"], [trainlines, vallines, testlines]):
        dataset = []
        for line in lines:
            text, example_triplets = parse_example(line)
            text, triplets_str = example_to_string(text, example_triplets)
            entry = {
                "text": text,
                "triplets": triplets_str,
                "triplets_str": get_extraction_aste_targets(
                    [line.split("####")[0].split()],
                    [eval(line.split("####")[1])]
                )[0],
            }
            dataset.append(entry)
        dataset_dict[name] = Dataset.from_list(dataset)
    dataset = DatasetDict(dataset_dict)
    dataset.save_to_disk(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--src", default='./data/bank_3200_sentenized', type=str, required=False)
    parser.add_argument("--dst", default='./data/bank_sentenized', type=str, required=False)
    args = parser.parse_args()
    main(args.src, args.dst)
