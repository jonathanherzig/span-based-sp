"""Creates GeoQuery Length split."""

from enum import Enum
import json


def read_data(input):
    data = []
    with open(input, "r") as data_file:
        for line_raw in data_file:
            line = line_raw.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line["question"]
            program = line["program"]
            data.append((question, program, line_raw, len(program.split())))
    return data


def write_split(examples, output):
    with open(output, 'w') as f:
        for line in examples:
            f.write(line)


class Version(Enum):
    REG = 0
    SPANS = 1


if __name__ == "__main__":

    version = Version.REG

    if version == Version.REG:
        train_input = 'datasets/geo/funql/train_orig.json'
        dev_input = 'datasets/geo/funql/test.json'
        train_output = 'datasets/geo/funql/train_orig_len.json'
        dev_output = 'datasets/geo/funql/test_len.json'
    else:
        train_input = 'data/geo/funql/train_spans.json'
        dev_input = 'data/geo/funql/dev_spans.json'
        train_output = 'data/geo/funql/train_len_spans.json'
        dev_output = 'data/geo/funql/dev_len_spans.json'

    data_train = read_data(train_input)
    data_dev = read_data(dev_input)
    data = data_train + data_dev
    data_sorted = sorted(data, key=lambda tup: tup[3])  # sort by program length

    train_len_examples = [ex[2] for ex in data_sorted[:600]]
    test_len_examples = [ex[2] for ex in data_sorted[600:]]

    write_split(train_len_examples, train_output)
    write_split(test_len_examples, dev_output)

