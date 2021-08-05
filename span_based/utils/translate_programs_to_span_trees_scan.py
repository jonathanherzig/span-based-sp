from collections import defaultdict
import json
from overrides import overrides
import traceback

from treelib import Tree

from span_based.utils.translate_programs_to_span_trees import SpanMapper
from span_based.utils.translate_span_trees_to_programs_scan import TreeMapperScan
from utils.executor_scan import ProgramExecutorScan

PREFIX = 'i_'
SUFFIX = ''


class SpanMapperScan(SpanMapper):

    def __init__(self):
        super().__init__()
        self._synonyms = {'arguments': defaultdict(set),
                          'predicates': defaultdict(list)}
        self._enrich_synonyms_by_hand()
        self._parser.setParseAction(self._parse_action)

    def _enrich_synonyms_by_hand(self):
        # {'predicates': ['jump', 'run', 'look', 'turn', 'walk'],
        #  'directions': ['right', 'left'],
        #  'manners': ['around', 'opposite'],
        #  'connectives': ['and', 'after'],
        #  'repetitions': ['twice', 'thrice']}
        self._synonyms['predicates'][PREFIX+'jump'+SUFFIX] = [self._get_wordpieces('jump')]
        self._synonyms['predicates'][PREFIX+'run'+SUFFIX] = [self._get_wordpieces('run')]
        self._synonyms['predicates'][PREFIX+'look'+SUFFIX] = [self._get_wordpieces('look')]
        self._synonyms['predicates'][PREFIX+'turn'+SUFFIX] = [self._get_wordpieces('turn')]
        self._synonyms['predicates'][PREFIX+'walk'+SUFFIX] = [self._get_wordpieces('walk')]
        self._synonyms['predicates'][PREFIX+'and'+SUFFIX] = [self._get_wordpieces('and')]
        self._synonyms['predicates'][PREFIX+'after'+SUFFIX] = [self._get_wordpieces('after')]
        self._synonyms['predicates'][PREFIX+'twice'+SUFFIX] = [self._get_wordpieces('twice')]
        self._synonyms['predicates'][PREFIX+'thrice'+SUFFIX] = [self._get_wordpieces('thrice')]
        self._synonyms['arguments'][PREFIX+'right'+SUFFIX] = set([self._get_wordpieces('right')])
        self._synonyms['arguments'][PREFIX+'left'+SUFFIX] = set([self._get_wordpieces('left')])
        self._synonyms['arguments'][PREFIX+'around'+SUFFIX] = set([self._get_wordpieces('around')])
        self._synonyms['arguments'][PREFIX+'opposite'+SUFFIX] = set([self._get_wordpieces('opposite')])


    @overrides
    def _parse_action(self, string, location, tokens) -> Tree:
        predicate = tokens[0][0]
        predicate_tree = self._get_tree_from_constant(predicate, 'predicates')
        len_args = len(tokens[0])
        if predicate in [PREFIX+'jump'+SUFFIX, PREFIX+'run'+SUFFIX, PREFIX+'look'+SUFFIX, PREFIX+'turn'+SUFFIX, PREFIX+'walk'+SUFFIX]:
            if len_args == 1:
                return predicate_tree
            else:
                arg_1_tree = self._get_tree_from_constant(tokens[0][1], 'arguments')
                if len_args == 2:
                    tree = self._join_unary_predicate_tree(predicate_tree, arg_1_tree)
                else:
                    arg_2_tree = self._get_tree_from_constant(tokens[0][2], 'arguments')
                    tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=arg_1_tree, arg_2=arg_2_tree)
                return tree
        elif predicate in [PREFIX+'twice'+SUFFIX, PREFIX+'thrice'+SUFFIX]:
            tree = self._join_unary_predicate_tree(predicate_tree, tokens[0][1])
            return tree
        elif predicate in [PREFIX+'and'+SUFFIX, PREFIX+'after'+SUFFIX]:
            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2])
            return tree
        else:
            raise Exception('unknown_predicate={}'.format(predicate))


if __name__ == "__main__":

    span_mapper = SpanMapperScan()
    tree_mapper = TreeMapperScan()
    executor = ProgramExecutorScan()

    # examples_path = ('data/scan/dsl/simple_split/test.json', 'data/scan/dsl/simple_split/test_spans.json')
    # examples_path = ('data/scan/dsl/simple_split/train.json', 'data/scan/dsl/simple_split/train_spans.json')
    # examples_path = ('data/scan/dsl/simple_split/test.json', 'data/scan/dsl/length_split/test_spans.json')
    # examples_path = ('data/scan/dsl/simple_split/train.json', 'data/scan/dsl/length_split/train_spans.json')
    # examples_path = ('data/scan/dsl/add_prim_split/test_turn_left.json', 'data/scan/dsl/add_prim_split/test_turn_left_spans.json')
    # examples_path = ('data/scan/dsl/add_prim_split/train_turn_left.json', 'data/scan/dsl/add_prim_split/train_turn_left_spans.json')
    # examples_path = ('data/scan/dsl/add_prim_split/test_jump.json', 'data/scan/dsl/add_prim_split/test_jump_spans.json')
    # examples_path = ('data/scan/dsl/add_prim_split/train_jump.json', 'data/scan/dsl/add_prim_split/train_jump_spans.json')
    # examples_path = ('data/scan/dsl/template_split/test_around_right.json', 'data/scan/dsl/template_split/test_around_right_spans.json')
    # examples_path = ('data/scan/dsl/template_split/train_around_right.json', 'data/scan/dsl/template_split/train_around_right_spans.json')
    # examples_path = ('data/scan/dsl/template_split/test_right.json', 'data/scan/dsl/template_split/test_right_spans.json')

    # examples_path = ('datasets/scan/dsl/train_simple.json', 'datasets/scan/dsl/train_simple_spans.json')
    # examples_path = ('datasets/scan/dsl/dev_simple.json', 'datasets/scan/dsl/dev_simple_spans.json')
    # examples_path = ('datasets/scan/dsl/test_simple.json', 'datasets/scan/dsl/test_simple_spans.json')
    # examples_path = ('datasets/scan/dsl/train_around_right.json', 'datasets/scan/dsl/train_around_right_spans.json')
    # examples_path = ('datasets/scan/dsl/dev_around_right.json', 'datasets/scan/dsl/dev_around_right_spans.json')
    # examples_path = ('datasets/scan/dsl/test_around_right.json', 'datasets/scan/dsl/test_around_right_spans.json')
    # examples_path = ('datasets/scan/dsl/train_right.json', 'datasets/scan/dsl/train_right_spans.json')
    # examples_path = ('datasets/scan/dsl/dev_right.json', 'datasets/scan/dsl/dev_right_spans.json')
    examples_path = ('datasets/scan/dsl/test_right.json', 'datasets/scan/dsl/test_right_spans.json')


    bad_unambiguous_trees = []
    bad_ambiguous_trees = []
    cannot_parse_trees = []
    good_parses = []
    back_translation_fail = []

    with open(examples_path[0], 'r') as data_file:
        for i, line in enumerate(data_file):
            # if i>=10000:
            #     break
            span_mapper._possible_chains = None
            span_mapper._possible_constants = {}
            if i % 100000 == 0:
                print('example {}'.format(i))
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question, program = line["question"], line["program"]
            print('example {}'.format(i))
            print(question)
            print(program)

            if i==319:
                print()

            # parse_tree_ = span_mapper.map_prog_to_tree(question, program)

            should_try_parse = True
            first_try = True
            while should_try_parse:
                parse_tree = None
                try:
                    parse_tree = span_mapper.map_prog_to_tree(question, program)
                # except ValueError as e:
                except Exception as e:
                    # except ValueError as e:
                    if first_try:
                        first_try = False
                        continue
                    cannot_parse_trees.append((question, program))
                    print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
                    print('cannot_parse: {} {} \n {}'.format(i, question, traceback.print_exc()))
                    print()
                    break

                is_valid = span_mapper.is_valid_tree(parse_tree)
                if not is_valid:
                    if first_try and (span_mapper._possible_chains or len(span_mapper._possible_constants) > 0):
                        first_try = False
                    elif first_try and not (
                    (span_mapper._possible_chains or len(span_mapper._possible_constants) > 0)):
                        bad_unambiguous_trees.append((parse_tree, question, program))
                        print('bad_trees={}'.format(len(bad_unambiguous_trees)))
                        should_try_parse = False
                        print()
                    else:
                        bad_ambiguous_trees.append((parse_tree, question, program))
                        print('bad_trees={}'.format(len(bad_ambiguous_trees)))
                        should_try_parse = False
                        print()
                else:
                    should_try_parse = False

            if parse_tree and is_valid:
                good_parses.append((line, parse_tree))
                program_bt = tree_mapper.map_tree_to_program(parse_tree)
                eq_bt = program == program_bt
                print(program)
                print(program_bt)
                print()
                if not eq_bt:
                    print()
                # if not eq_bt:
                #     denotation = executor.execute(program)
                #     denotation_bt = executor.execute(program_bt)
                #     if denotation != denotation_bt:
                #         back_translation_fail.append((question, program, program_bt))
                # # if not eq_bt and not program.startswith('query_attribute_equal') and not program.startswith(
                # #         'count_equal'):
                # #     raise Exception('Bad backtranslation failure={} \n {}'.format(question, program))
            # if parse_tree and not is_valid and not first_try:
            #     print('here')
            # if parse_tree and not is_valid:
            #     print('here')

    print('bad_unambiguous_trees={}'.format(len(bad_unambiguous_trees)))
    print('bad_ambiguous_trees={}'.format(len(bad_ambiguous_trees)))
    print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
    print('back_translation_fail={}'.format(len(back_translation_fail)))
    print('total_parsed={}'.format(i+1))

    with open(examples_path[1], 'w') as output_file:
        for good_parse in good_parses:
            span_mapper.write_to_output(line=good_parse[0], parse_tree=good_parse[1], output_file=output_file)


    # for fail in back_translation_fail:
    #     print('ques = {}'.format(fail[0]))
    #     print('prog = {}'.format(fail[1]))
    #     print('prbt = {}'.format(fail[2]))
    #     print()

    # for ex in bad_unambiguous_trees:
    #     print('{} {}'.format(ex[1], ex[2]))
