"""Processes GeoQuery in original format to the json format we work with."""

import os
import json


def read_examples(input):
    """Reads original FunQL file."""

    ids = []
    questions = []
    programs = []

    with open(input, 'r') as f:
        for row in f:
            if row.startswith('id:'):
                id = row.replace('id:','').strip()
                ids.append(id)
            elif row.startswith('nl:'):
                question = row.replace('nl:', '').replace(" 's ", "'s ").strip()
                questions.append(question)
            elif row.startswith('mrl:'):
                program = row.replace('mrl:', '').replace(')', ' ) ').replace('(', ' ( ').replace('  ', ' ').strip()
                programs.append(program)

    assert len(ids) == len(questions) == len(programs)
    return ids, questions, programs


def read_train_ids(train_ids_input):

    with open(train_ids_input, 'r') as f:
        train_ids = f.readlines()
        train_ids = [id.strip() for id in train_ids]
        return train_ids


def write_question_split(ids, questions, programs, train_ids, out_path):

    with open(os.path.join(out_path, 'train_orig' + '.json'), 'w') as f_train:
        with open(os.path.join(out_path, 'test' + '.json'), 'w') as f_dev:
            for id, question, program in zip(ids, questions, programs):
                if id in train_ids:
                    json.dump({'question': question, 'program': program}, f_train)
                    f_train.write('\n')
                else:
                    json.dump({'question': question, 'program': program}, f_dev)
                    f_dev.write('\n')


if __name__ == "__main__":
    input = 'datasets/geo/raw/geoFunql-en.corpus'
    train_ids_input = 'datasets/geo/raw/split880.train.ids'
    out_path = 'datasets/geo/funql/'
    ids, questions, programs = read_examples(input)
    train_ids = read_train_ids(train_ids_input)
    write_question_split(ids, questions, programs, train_ids, out_path)