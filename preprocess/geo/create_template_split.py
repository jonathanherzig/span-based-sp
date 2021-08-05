"""Creates GeoQuery Template split."""

from enum import Enum
import json
import random
import re
from collections import defaultdict
import matplotlib.pyplot as plt


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
            template = re.sub('stateid \( .*? \)', 'stateid ( _state_ )', program)
            template = re.sub('cityid \( .*? \)', 'cityid ( _city_ )', template)
            template = re.sub('countryid \( .*? \)', 'countryid ( _country_ )', template)
            template = re.sub('riverid \( .*? \)', 'riverid ( _river_ )', template)
            template = re.sub('placeid \( .*? \)', 'placeid ( _place_ )', template)
            print(program)
            print(template)
            print()
            data.append((question, program, template, line_raw))
    return data


def make_template_split(templates, len_train):

    train_templated_examples = []
    dev_templated_examples = []
    train_freqs = []
    dev_freqs = []

    for template in templates:
        if len(train_templated_examples) < len_train:
            train_freqs.append(len(templates[template]))
            for example in templates[template]:
                train_templated_examples.append(example[3])
        else:
            dev_freqs.append(len(templates[template]))
            for example in templates[template]:
                dev_templated_examples.append(example[3])
    return train_templated_examples, dev_templated_examples, train_freqs, dev_freqs


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
        train_output = 'datasets/geo/funql/train_orig_template.json'
        dev_output = 'datasets/geo/funql/test_template.json'
    else:
        train_input = 'data/geo/funql/train_spans.json'
        dev_input = 'data/geo/funql/dev_spans.json'
        train_output = 'data/geo/funql/train_template_spans.json'
        dev_output = 'data/geo/funql/dev_template_spans.json'

    data_train = read_data(train_input)
    data_dev = read_data(dev_input)
    data = data_train + data_dev

    templates = defaultdict(list)
    for example in data:
        templates[example[2]].append(example)
    print('total_templates={}'.format(len(templates)))

    # randomize templates
    random.seed(10)
    keys = list(templates.keys())
    random.shuffle(keys)
    templates_shuffled = dict()
    for key in keys:
        templates_shuffled.update({key: templates[key]})
    templates = templates_shuffled

    template_freq = [len(templates[template]) for template in templates]
    plt.hist(template_freq, bins=20, range=(-1,20))
    plt.show()

    train_templated_examples, dev_templated_examples, train_freqs, dev_freqs = make_template_split(templates, len(data_train))
    print('train_templated_examples={}'.format(len(train_templated_examples)))
    print('dev_templated_examples={}'.format(len(dev_templated_examples)))

    plt.hist(train_freqs, bins=20, range=(-1, 20))
    plt.show()
    plt.hist(dev_freqs, bins=20, range=(-1, 20))
    plt.show()

    write_split(train_templated_examples, train_output)
    write_split(dev_templated_examples, dev_output)

