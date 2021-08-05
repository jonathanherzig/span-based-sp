"""Converts SCAN to a semantic parsing task, named SCAN-SP."""

from enum import Enum
import json
import math
import random

PREFIX = 'i_'
SUFFIX = ''


def convert_to_lf(input: str, output_1: str, output_2: str = None):
    """
    Converts a single input file in original SCAN formalism to SCAN-SP formalism.

    # Parameters
    input : path to input file in original SCAN formalism.
    output_1 : path to training file output.
    output_2 : path to dev file output.
    """

    commands = {'predicates': ['jump', 'run', 'look', 'turn', 'walk'],
                'directions': ['right', 'left'],
                'manners': ['around', 'opposite'],
                'connectives': ['and', 'after'],
                'repetitions':  ['twice', 'thrice']}

    examples_parsed = []

    all_possible_tokens = []
    for tokens in commands.values():
        all_possible_tokens += tokens
    with open(input, 'r') as f:
        for row in f:
            connective = None
            question = row.split('OUT:')[0].replace('IN:', '').strip()
            denotation = row.split('OUT:')[1].strip()
            parts = [question]
            for token in parts[0].split(' '):
                assert token in all_possible_tokens
            for connective_candidate in commands['connectives']:
                parts = parts[0].split(connective_candidate)
                if len(parts) > 1:
                    connective = connective_candidate
                    break
            inner_programs = []
            for i, part in enumerate(parts):
                inner_programs.append(get_inner_program(part.split(' '), commands))
            if not connective:
                assert len(inner_programs) == 1
                program = inner_programs[0]
            else:
                assert len(inner_programs) == 2
                program = '{} ( {} , {} )'.format(PREFIX+connective+SUFFIX, inner_programs[0],
                                                  inner_programs[1])
            program = program.replace('  ', ' ')
            examples_parsed.append({'question': question, 'program': program, 'answer': denotation})
    if output_2 is not None:  # take 20% for dev
        random.shuffle(examples_parsed)
        train_size = math.ceil(0.8 * len(examples_parsed))
        with open(output_1, 'w') as f_1:
            with open(output_2, 'w') as f_2:
                for i, ex in enumerate(examples_parsed):
                    if i < train_size:
                        json.dump(ex, f_1)
                        f_1.write('\n')
                    else:
                        json.dump(ex, f_2)
                        f_2.write('\n')
    else:
        with open(output_1, 'w') as f_1:
            for i, ex in enumerate(examples_parsed):
                    json.dump(ex, f_1)
                    f_1.write('\n')


def get_inner_program(tokens, commands):
    inner_program = []
    for repetition in commands['repetitions']:
        if repetition in tokens:
            inner_program.append(PREFIX+repetition+SUFFIX)
            break
    most_iner = []
    for predicate in commands['predicates']:
        if predicate in tokens:
            most_iner.append(PREFIX+predicate+SUFFIX)
            break
    for direction in commands['directions']:
        if direction in tokens:
            most_iner.append(PREFIX+direction+SUFFIX)
            break
    for manner in commands['manners']:
        if manner in tokens:
            most_iner.append(PREFIX+manner+SUFFIX)
            break
    most_iner_str = '{} ( {} )'.format(most_iner[0], ' , '.join(most_iner[1:]))
    if not inner_program:
        return most_iner_str
    else:
        return '{} ( {} )'.format(inner_program[0], most_iner_str)


class Split(Enum):
    SIMPLE = 0
    RIGHT = 1
    AROUND_RIGHT = 2


if __name__ == "__main__":

    split = Split.SIMPLE

    paths = {
             split.SIMPLE: {"inputs": ['datasets/scan/simple_split/tasks_train_simple.txt',
                                       'datasets/scan/simple_split/tasks_test_simple.txt'],
                            "outputs": ['datasets/scan/dsl/train_simple.json',
                                        'datasets/scan/dsl/dev_simple.json',
                                        'datasets/scan/dsl/test_simple.json']},
             Split.AROUND_RIGHT: {
                 "inputs": ['datasets/scan/template_split/tasks_train_template_around_right.txt',
                            'datasets/scan/template_split/tasks_test_template_around_right.txt'],
                 "outputs": ['datasets/scan/dsl/train_around_right.json',
                             'datasets/scan/dsl/dev_around_right.json',
                             'datasets/scan/dsl/test_around_right.json']},
             split.RIGHT: {
                 "inputs": ['datasets/scan/template_split/tasks_train_template_right.txt',
                            'datasets/scan/template_split/tasks_test_template_right.txt'],
                 "outputs": ['datasets/scan/dsl/train_right.json',
                             'datasets/scan/dsl/dev_right.json',
                             'datasets/scan/dsl/test_right.json']},
             }

    random.seed(0)

    convert_to_lf(paths[split]["inputs"][0], paths[split]["outputs"][0],
                  output_2=paths[split]["outputs"][1])
    convert_to_lf(paths[split]["inputs"][1], paths[split]["outputs"][2], output_2=None)
