from collections import defaultdict
import json
from overrides import overrides
import traceback

from span_based.utils.translate_span_trees_to_programs_clevr import TreeMapperClevr
from span_based.utils.translate_programs_to_span_trees import SpanMapper
from treelib import Tree
from utils.executor_clevr import ProgramExecutorClevr


class SpanMapperClevr(SpanMapper):

    def __init__(self):
        super().__init__()
        self._metadata_path = 'data/clevr/orig/question_generation/metadata.json'
        self._synonyms_path = 'data/clevr/orig/question_generation/synonyms.json'
        self._load_argumets()
        self._enrich_synonyms()
        self._enrich_synonyms_by_hand()
        self._parser.setParseAction(self._parse_action)

    def _load_argumets(self):
        with open(self._metadata_path, 'r') as json_file:
            data = json.load(json_file)
        for type in data['types']:
            if data['types'][type] is not None:
                for value in data['types'][type]:
                    self._synonyms['arguments'][value] = set([self._get_wordpieces(value)])

    def _enrich_synonyms(self):
        with open(self._synonyms_path, 'r') as json_file:
            data = json.load(json_file)
        for argument in data:
            if argument not in self._synonyms['arguments']:
                self._synonyms['arguments'][argument] = set()
            for realization in data[argument]:
                self._synonyms['arguments'][argument].update(set([self._get_wordpieces(realization)]))

    def _enrich_synonyms_by_hand(self):
        self._synonyms['predicates']['count_greater'] = set([self._get_wordpieces('are there more'),
                                                             self._get_wordpieces('greater than')])

        self._synonyms['predicates']['count_less'] = set([self._get_wordpieces('are there fewer'),
                                                          self._get_wordpieces('less than')
                                                        ])

        self._synonyms['predicates']['count'] = set([self._get_wordpieces('how many'),
                                                     self._get_wordpieces('what number')])

        self._synonyms['predicates']['relate_attribute_equal'] = set([self._get_wordpieces('of the same'),
                                                                      self._get_wordpieces('is the same'),
                                                                      self._get_wordpieces('that have the same'),
                                                                      self._get_wordpieces('are the same'),
                                                                      self._get_wordpieces('that has the same'),
                                                                      self._get_wordpieces('have the same')])


        self._synonyms['predicates']['query_attribute_equal'] = set([self._get_wordpieces('the same')])

        self._synonyms['predicates']['query'] = set([self._get_wordpieces('what'),
                                                     self._get_wordpieces('how'),
                                                     self._get_wordpieces('what is'),
                                                     self._get_wordpieces('is what')])

        self._synonyms['predicates']['intersect'] = set([self._get_wordpieces('and')])

        self._synonyms['predicates']['union'] = set([self._get_wordpieces('or')])

        self._synonyms['predicates']['exist'] = set([self._get_wordpieces('are there'),
                                                     self._get_wordpieces('are any'),
                                                     self._get_wordpieces('is there')])

        self._synonyms['predicates']['count_equal'] = set([self._get_wordpieces('are there the same number'),
                                                           self._get_wordpieces('the same as the number'),
                                                           self._get_wordpieces('are there an equal number')])


        self._synonyms['arguments']['shape'] = set([self._get_wordpieces('shape')])
        self._synonyms['arguments']['color'] = set([self._get_wordpieces('color')])
        self._synonyms['arguments']['material'] = set([self._get_wordpieces('material'),
                                                      self._get_wordpieces('made of')])
        self._synonyms['arguments']['size'] = set([self._get_wordpieces('big'),
                                                   self._get_wordpieces('size')])

        self._synonyms['arguments']['sphere'].update([self._get_wordpieces('spheres'), self._get_wordpieces('balls')])
        self._synonyms['arguments']['cylinder'].add(self._get_wordpieces('cylinders'))
        self._synonyms['arguments']['cube'].add(self._get_wordpieces('blocks'))

        # self._synonyms['predicates']['count_greater'] = set([self._get_wordpieces('more'),
        #                                                      self._get_wordpieces('greater than')])
        #
        # self._synonyms['predicates']['count_less'] = set([self._get_wordpieces('fewer'),
        #                                                   self._get_wordpieces('less')
        #                                                   ])
        #
        # self._synonyms['predicates']['count'] = set([self._get_wordpieces('how many'),
        #                                              self._get_wordpieces('what number')])
        #
        # self._synonyms['predicates']['relate_attribute_equal'] = set(
        #     [self._get_wordpieces('same'),])
        #
        # self._synonyms['predicates']['query_attribute_equal'] = set(
        #     [self._get_wordpieces('same')])
        #
        # self._synonyms['predicates']['query'] = set([self._get_wordpieces('what'),
        #                                              self._get_wordpieces('how'),])
        #
        # self._synonyms['predicates']['intersect'] = set([self._get_wordpieces('and')])
        #
        # self._synonyms['predicates']['union'] = set([self._get_wordpieces('or')])
        #
        # self._synonyms['predicates']['exist'] = set([self._get_wordpieces('there'),
        #                                              self._get_wordpieces('any'),])
        #
        # self._synonyms['predicates']['count_equal'] = set(
        #     [self._get_wordpieces('same number'),
        #      self._get_wordpieces('the same as the number'),
        #      self._get_wordpieces('equal number')])
        #
        # self._synonyms['arguments']['shape'] = set([self._get_wordpieces('shape')])
        # self._synonyms['arguments']['color'] = set([self._get_wordpieces('color')])
        # self._synonyms['arguments']['material'] = set([self._get_wordpieces('material'),
        #                                                self._get_wordpieces('made of')])
        # self._synonyms['arguments']['size'] = set([self._get_wordpieces('big'),
        #                                            self._get_wordpieces('size')])
        #
        # self._synonyms['arguments']['sphere'].update(
        #     [self._get_wordpieces('spheres'), self._get_wordpieces('balls')])
        # self._synonyms['arguments']['cylinder'].add(self._get_wordpieces('cylinders'))
        # self._synonyms['arguments']['cube'].add(self._get_wordpieces('blocks'))

        filter_args = ['cube', 'sphere', 'cylinder', 'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow',
                       'small', 'large', 'rubber', 'metal']
        self._filter_args = [arg_instance for instances in [self._synonyms['arguments'][arg] for arg in filter_args] for arg_instance in instances]

    def _collect_freq(self, tree, pred_to_freq):
        for key, node in tree.nodes.items():  # save freq
            constant = node.data.constant
            content = node.data.content
            if constant is not None:
                if constant not in pred_to_freq:
                    pred_to_freq[constant] = defaultdict(int)
                pred_to_freq[constant][content] += 1

    @overrides
    def _parse_action(self, string, location, tokens) -> Tree:
        from collections import namedtuple
        Node = namedtuple("Node", ["value", "children"])
        predicate = tokens[0][0]
        if predicate == 'scene':
            return 'scene'
        elif predicate == 'filter':
            args = tokens[0][1: -1]
            world = tokens[0][-1]
            hint_span = world.get_node(world.root).data if world != 'scene' else None
            sub_tree = self._align_filter_to_text(args, type='arguments', hint_span=hint_span)

            self._collect_freq(sub_tree, pred_to_freq)

            if world != 'scene':
                print('here')
                sub_tree = self._join_unary_predicate_tree(sub_tree, world)
            return sub_tree
        elif predicate == 'relate_attribute_equal':
            attribute = tokens[0][1]
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')
            hint_span = self._get_aligned_span(predicate_tree)
            attribute_tree = self._get_tree_from_constant(attribute, 'arguments', hint_span=hint_span)

            self._collect_freq(predicate_tree, pred_to_freq)
            self._collect_freq(attribute_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=attribute_tree, arg_2=tokens[0][2])
            return tree
        elif predicate == 'exist':
            argument_tree = tokens[0][1]
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_unary_predicate_tree(predicate_tree, argument_tree)
            return tree
        elif predicate == 'query':
            attribute = tokens[0][1]
            argument_tree = tokens[0][2]
            hint_span = self._get_aligned_span(argument_tree)
            attribute_tree = self._get_tree_from_constant(attribute, 'arguments', hint_span)
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)
            self._collect_freq(attribute_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=attribute_tree, arg_2=argument_tree)
            return tree
        elif predicate == 'relate':
            relation = tokens[0][1]
            argument_tree = tokens[0][2]
            hint_span = self._get_aligned_span(argument_tree)
            relation_tree = self._get_tree_from_constant(relation, 'arguments', hint_span)

            self._collect_freq(relation_tree, pred_to_freq)

            tree = self._join_unary_predicate_tree(relation_tree, argument_tree)
            return tree
        elif predicate == 'count':
            argument_tree = tokens[0][1]
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_unary_predicate_tree(predicate_tree, argument_tree)
            return tree
        elif predicate == 'query_attribute_equal':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')
            attribute = tokens[0][1]
            arg_tree_1 = tokens[0][2]
            arg_tree_2 = tokens[0][3]
            fist_arg, second_arg = self._get_first_argument_to_join(predicate_tree, arg_tree_1, arg_tree_2)
            attribute_tree = self._get_tree_from_constant(attribute, 'arguments', hint_span=self._get_aligned_span(predicate_tree))
            # binary_join = self._join_binary_predicate_tree(predicate_tree, attribute_tree, arg_tree_1)
            # tree = self._join_unary_predicate_tree(binary_join, arg_tree_2)

            self._collect_freq(predicate_tree, pred_to_freq)
            self._collect_freq(attribute_tree, pred_to_freq)

            binary_join = self._join_binary_predicate_tree(predicate_tree, attribute_tree, fist_arg)
            tree = self._join_unary_predicate_tree(binary_join, second_arg)
            return tree
        elif predicate == 'union':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2])
            return tree
        elif predicate == 'intersect':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2])
            return tree
        elif predicate == 'count_greater':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2], allow_arg_switch=False)
            return tree

        elif predicate == 'count_less':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2], allow_arg_switch=False)
            return tree
        elif predicate == 'count_equal':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')

            self._collect_freq(predicate_tree, pred_to_freq)

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2])
            return tree
        else:
            raise Exception('unknown_predicate={}'.format(predicate))


if __name__ == "__main__":

    span_mapper = SpanMapperClevr()
    tree_mapper = TreeMapperClevr()
    executor = ProgramExecutorClevr()

    examples_path = ('datasets/clevr/dsl/dev.json', 'datasets/clevr/dsl/dev_spans.json')
    # examples_path = ('datasets/clevr/dsl/train_small.json', 'datasets/clevr/dsl/train_small_spans.json')
    # examples_path = ('datasets/clevr/dsl/train.json', 'datasets/clevr/dsl/train_spans.json')
    # examples_path = ('data/clevr/dsl/closure_val.json', 'data/clevr/dsl/closure_val_spans.json')
    # examples_path = ('datasets/clevr/dsl/test.json', 'datasets/clevr/dsl/test_spans.json')
    # examples_path = ('datasets/clevr/dsl/closure.json', 'datasets/clevr/dsl/closure_spans.json')
    # start_time = time.time()

    bad_unambiguous_trees = []
    bad_ambiguous_trees = []
    cannot_parse_trees = []
    good_parses = []
    back_translation_fail = []

    pred_to_freq = {}

    with open(examples_path[0], 'r') as data_file:
        for i, line in enumerate(data_file):
            # if i>=10000:
            #     break
            span_mapper._possible_chains = None
            span_mapper._possible_constants = {}
            # if i==123:
            #     print('here')
            # if i in [136, 246, 269, 339, 387]:
            #     cannot_parse_trees.append(question)
            #     print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
            #     continue
            if i%100000==0:
                print('example {}'.format(i))
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question, program, answer, scene = line["question"], line["program"], line["answer"], line["scene"]
            print('example {}'.format(i))
            print(question)
            print(program)
            print(scene)
            print(answer)
            # if question == 'Are the small sphere and the green block made of the same material?':
            #     print('here')
            # if question == 'There is a small gray metal ball; how many gray balls are on the right side of it?':
            #     print('here')
            # if i == 6:
            #     print('here')

            # question = "Are the yellow cylinder to the left of the block and the tiny yellow thing made of the same material?"
            # program= "query_attribute_equal ( material , filter ( yellow , cylinder , relate ( left , filter ( cube , scene (  ) ) ) ) , filter ( small , yellow , scene (  ) ) )"

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
                    cannot_parse_trees.append(question)
                    print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
                    print('cannot_parse: {} {} \n {}'.format(i, question, traceback.print_exc()))
                    print()
                    break

                is_valid = span_mapper.is_valid_tree(parse_tree)
                if not is_valid:
                    if first_try and (span_mapper._possible_chains or len(span_mapper._possible_constants)>0):
                        first_try = False
                    elif first_try and not ((span_mapper._possible_chains or len(span_mapper._possible_constants)>0)):
                        bad_unambiguous_trees.append((parse_tree, question))
                        print('bad_trees={}'.format(len(bad_unambiguous_trees)))
                        should_try_parse = False
                        print()
                    else:
                        bad_ambiguous_trees.append((parse_tree, question))
                        print('bad_trees={}'.format(len(bad_ambiguous_trees)))
                        should_try_parse = False
                        print()
                else:
                    should_try_parse = False

            if parse_tree and is_valid:
                good_parses.append((line, parse_tree))
                program_bt = tree_mapper.map_tree_to_program(parse_tree)
                program_ = program.replace('  ',' ')
                eq_bt = program_ == program_bt
                print(program)
                print(program_bt)
                if not eq_bt:
                    denotation = executor.execute(program, scene)
                    denotation_bt = executor.execute(program_bt, scene)
                    if denotation != denotation_bt:
                        back_translation_fail.append((question, program, program_bt))
                if not eq_bt and not program.startswith('query_attribute_equal') and not program.startswith('count_equal'):
                    raise Exception('Bad backtranslation failure={} \n {}'.format(question, program))
            if parse_tree and not is_valid and not first_try:
                print('here')
            if parse_tree and not is_valid:
                print('here')

    with open(examples_path[1], 'w') as output_file:
        for good_parse in good_parses:
            span_mapper.write_to_output(line=good_parse[0], parse_tree=good_parse[1], output_file=output_file)

    print('bad_unambiguous_trees={}'.format(len(bad_unambiguous_trees)))
    print('bad_ambiguous_trees={}'.format(len(bad_ambiguous_trees)))
    print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
    print('back_translation_fail={}'.format(len(back_translation_fail)))
    print('total_parsed={}'.format(i))

    for fail in back_translation_fail:
        print('ques = {}'.format(fail[0]))
        print('prog = {}'.format(fail[1]))
        print('prbt = {}'.format(fail[2]))
        print()

    # for ex in bad_ambiguous_trees:
    #     print(ex)

    for pred in pred_to_freq:
        d = pred_to_freq[pred]
        for k in sorted(d, key=d.get, reverse=True):
            print('{} {} {}'.format(pred,k,d[k]))


    # # prepare a split by tree depth
    # import matplotlib.pyplot as plt
    # dephts = [tree[1].depth() for tree in good_parses]
    # plt.hist(dephts, bins=20, range=(-1,20))
    # plt.show()
    #
    # train_path_depths = 'data/clevr/dsl/train_spans_depths.json'
    # dev_path_depths = 'data/clevr/dsl/val_spans_depths.json'
    # with open(train_path_depths, 'w') as f_train:
    #     with open(dev_path_depths, 'w') as f_dev:
    #         for good_parse in good_parses:
    #             if good_parse[1].depth()<=2:
    #                 write_to_output(line=good_parse[0], parse_tree=good_parse[1], output_file=f_train,
    #                                 tokenizer=span_mapper._get_wordpieces)
    #             else:
    #                 if good_parse[1].depth()>5:
    #                     write_to_output(line=good_parse[0], parse_tree=good_parse[1], output_file=f_dev,
    #                                     tokenizer=span_mapper._get_wordpieces)
