"""Splits training data to train and development (in case there is no natural development set)."""

from enum import Enum
import json
import random


def split_train(top_to_train: int, bottom_to_dev: int, input: str, output_train: str,
                output_dev: str, shuffle: bool):
    """Takes top_to_train examples for training and bottom_to_dev for development."""

    examples = []
    with open(input, 'r') as data_file:
        for line in data_file:
            line = line.strip('\n')
            line = json.loads(line)
            examples.append(line)

    if shuffle:
        random.shuffle(examples)

    dev_start = (len(examples) - bottom_to_dev)
    with open(output_train, 'w') as f_train:
        with open(output_dev, 'w') as f_dev:
            for i, ex in enumerate(examples):
                if (top_to_train is None and i < dev_start) or (
                        top_to_train is not None and i < top_to_train):
                    json.dump(ex, f_train)
                    f_train.write('\n')
                elif i >= dev_start:
                    json.dump(ex, f_dev)
                    f_dev.write('\n')


class Version(Enum):
    CLEVR = 0
    GEO = 1
    GEO_TEMPLATE = 2
    GEO_LEN = 3


if __name__ == '__main__':

    random.seed(0)
    version = Version.GEO_LEN

    if version == Version.CLEVR:
        input = 'datasets/clevr/dsl/train_orig.json'
        output_train = 'datasets/clevr/dsl/train_small.json'
        output_dev = 'datasets/clevr/dsl/dev.json'
        split_train(10000, 5000, input, output_train, output_dev, shuffle=False)
    elif version == Version.GEO:
        input = 'datasets/geo/funql/train_orig.json'
        output_train = 'datasets/geo/funql/train.json'
        output_dev = 'datasets/geo/funql/dev.json'
        split_train(540, 60, input, output_train, output_dev, shuffle=True)
    elif version == Version.GEO_TEMPLATE:
        input = 'datasets/geo/funql/train_orig_template.json'
        output_train = 'datasets/geo/funql/train_template.json'
        output_dev = 'datasets/geo/funql/dev_template.json'
        split_train(544, 60, input, output_train, output_dev, shuffle=True)
    else:
        input = 'datasets/geo/funql/train_orig_len.json'
        output_train = 'datasets/geo/funql/train_len.json'
        output_dev = 'datasets/geo/funql/dev_len.json'
        split_train(540, 60, input, output_train, output_dev, shuffle=True)
