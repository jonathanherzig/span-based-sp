"""Processes CLEVR and CLOSURE into DSL programs (requires downloading original CLEVR data)."""

from copy import deepcopy
from collections import Counter
from enum import Enum
import json
from typing import List, Text


def orig_to_dsl(input_paths, output_path, image_to_scene):
    """Parses an original CLEVR program into the DSL we work with."""

    examples = []
    answer_counter = Counter()
    for input_path in input_paths:
        with open(input_path, 'rb') as f:
            data = json.load(f)
            for i, question_details in enumerate(data['questions']):
                example = {}
                question = question_details['question']
                scene = image_to_scene[question_details['image_filename'].replace('.png', '')]
                scene_str = '***'.join(scene)

                try:
                    id = question_details['question_index']
                    example['id'] = id
                except:
                    print('no question_index')
                example['question'] = question
                try:
                    program_intermediate = clevr_to_nsclseq(question_details['program'])
                    program_tree = nsclseq_to_nscltree(program_intermediate)
                    program = parse_program_tree(program_tree)
                    example['program'] = program
                except:
                    print('no program')
                if 'answer' in question_details:
                    answer = question_details['answer']
                    example['answer'] = answer
                    answer_counter.update([answer])
                example['scene'] = scene_str

                examples.append(example)

    print('answer_counter={}'.format(answer_counter))
    total = sum(answer_counter.values())
    freq = {}
    for a in answer_counter:
        freq[a] = float(answer_counter[a]) / total
    print(freq)

    with open(output_path, 'w') as f:
        for i, example in enumerate(examples):
            if i % 1000 == 0:
                print('written {} examples'.format(i))
            json.dump(example, f)
            f.write('\n')


def clevr_to_nsclseq(clevr_program):
    nscl_program = list()
    mapping = dict()

    for block_id, block in enumerate(clevr_program):
        op = get_clevr_pblock_op(block)
        current = None
        if op == 'scene':
            current = dict(op='scene')
        elif op.startswith('filter'):
            concept = block['value_inputs'][0]
            last = nscl_program[mapping[block['inputs'][0]]]
            if last['op'] == 'filter':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif op.startswith('relate'):
            concept = block['value_inputs'][0]
            current = dict(op='relate', relational_concept=[concept])
        elif op.startswith('same'):
            attribute = get_clevr_op_attribute(op)
            current = dict(op='relate_attribute_equal', attribute=attribute)
        elif op in ('intersect', 'union'):
            current = dict(op=op)
        elif op == 'unique':
            pass  # We will ignore the unique operations.
        else:
            if op.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevr_op_attribute(op)
                    current = dict(op='query', attribute=attribute)
            elif op.startswith('equal') and op != 'equal_integer':
                attribute = get_clevr_op_attribute(op)
                current = dict(op='query_attribute_equal', attribute=attribute)
            elif op == 'exist':
                current = dict(op='exist')
            elif op == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            elif op == 'equal_integer':
                current = dict(op='count_equal')
            elif op == 'less_than':
                current = dict(op='count_less')
            elif op == 'greater_than':
                current = dict(op='count_greater')
            else:
                raise ValueError('Unknown CLEVR operation: {}.'.format(op))

        if current is None:
            assert len(block['inputs']) == 1
            mapping[block_id] = mapping[block['inputs'][0]]
        else:
            current['inputs'] = list(map(mapping.get, block['inputs']))

            if '_output' in block:
                current['output'] = deepcopy(block['_output'])

            nscl_program.append(current)
            mapping[block_id] = len(nscl_program) - 1

    return nscl_program


def parse_program_tree(program_tree):

    def parse_node(node):
        num_inputs = len(node['inputs'])
        concept = ' , '.join(node['concept'])+' , ' if 'concept' in node else ''
        attribute = node['attribute']+' , ' if 'attribute' in node else ''
        relational_concept = ' , '.join(node['relational_concept'])+' , ' if 'relational_concept' in node else ''
        assert(not (len(concept) > 0 and len(attribute) > 0))
        if num_inputs == 0:
            return '{} ( {}{}{} )'.format(node['op'], concept, attribute, relational_concept)
        elif num_inputs == 1:
            return '{} ( {}{}{}{} )'.format(node['op'], concept, attribute, relational_concept, parse_node(node['inputs'][0]))
        elif num_inputs == 2:
            return '{} ( {}{}{}{} , {} )'.format(node['op'], concept, attribute, relational_concept, parse_node(node['inputs'][0]), parse_node(node['inputs'][1]))
        else:
            assert(False)

    return parse_node(program_tree)


def nsclseq_to_nscltree(seq_program):
    def dfs(sblock):
        tblock = deepcopy(sblock)
        input_ids = tblock.pop('inputs')
        tblock['inputs'] = [dfs(seq_program[i]) for i in input_ids]
        return tblock
    try:
        return dfs(seq_program[-1])
    finally:
        del dfs


def get_clevr_pblock_op(block):
    """
    Return the operation of a CLEVR program block.
    """
    if 'type' in block:
        return block['type']
    assert 'function' in block
    return block['function']


def get_clevr_op_attribute(op):
    return op.split('_')[1]


def get_object_relative_index(scene_details, id, relation):
    """ Gets the relative position of an object in the scene."""
    relative_index = len(scene_details['relationships'][relation][id])
    return str(relative_index)


def read_scenes(files: List[Text]):
    """Parses scene files into a String that describes it."""
    image_to_scene = {}
    for file in files:
        with open(file, 'rb') as f:
            data = json.load(f)
            for i, scene_details in enumerate(data['scenes']):
                image_index = scene_details['image_filename'].replace('.png','')
                objects = ['{} {} {} {} {} {}'.format(obj['shape'], obj['material'], obj['size'], obj['color'],
                                                   get_object_relative_index(scene_details, i, 'left'),
                                                   get_object_relative_index(scene_details, i, 'front'))
                           for i, obj in enumerate(scene_details['objects'])]
                image_to_scene[image_index] = objects
    return image_to_scene


class Version(Enum):
    CLEVR = 0
    CLOSURE = 1


if __name__ == "__main__":
    # scene paths
    train_scene_path = 'data/clevr/orig/scenes/CLEVR_train_scenes.json'
    val_scene_path = 'data/clevr/orig/scenes/CLEVR_val_scenes.json'

    image_to_scene = read_scenes([train_scene_path, val_scene_path])

    version = Version.CLEVR

    if version == Version.CLEVR:
        for split in ['train', 'val']:
            input_paths = ['data/clevr/orig/questions/CLEVR_{}_questions.json'.format(split)]
            # We take the original val set as our test set.
            out_split_name = 'test' if split == 'val' else 'train_orig'
            output_path = 'datasets/clevr/dsl/{}.json'.format(out_split_name)
            orig_to_dsl(input_paths, output_path, image_to_scene)
    elif version == Version.CLOSURE:
        splits = 'and_mat_spa_val', 'compare_mat_spa_val', 'compare_mat_val', 'embed_mat_spa_val', 'embed_spa_mat_val', 'or_mat_spa_val', 'or_mat_val'
        input_paths = ['data/clevr/CLOSURE-master/{}.json'.format(split) for split in splits]
        output_path = 'datasets/clevr/dsl/{}.json'.format('closure')
        orig_to_dsl(input_paths, output_path, image_to_scene)
