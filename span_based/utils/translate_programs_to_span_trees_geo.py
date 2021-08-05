from collections import defaultdict
import json
from overrides import overrides
import traceback

from treelib import Tree

from span_based.utils.translate_programs_to_span_trees import SpanMapper
from span_based.utils.translate_span_trees_to_programs_geo import TreeMapperGeo
from utils.geo_eval.executor_geo import ProgramExecutorGeo


class SpanMapperGeo(SpanMapper):

    def __init__(self):
        super().__init__()
        self._synonyms = {'arguments': defaultdict(set),
                          'predicates': defaultdict(list)}
        self._enrich_synonyms_by_hand()
        self._parser.setParseAction(self._parse_action)

    def _enrich_synonyms_by_hand(self):
        self._synonyms['predicates']['loc_2'] = [self._get_wordpieces('in'),
                                                 self._get_wordpieces('of'),
                                                 self._get_wordpieces('have'),
                                                 self._get_wordpieces('has')]
        self._synonyms['predicates']['loc_1'] = [self._get_wordpieces('with'),
                                                 self._get_wordpieces('whose'),
                                                 self._get_wordpieces('have'),
                                                 self._get_wordpieces('in'),
                                                 self._get_wordpieces('has'),
                                                 self._get_wordpieces('of'),
                                                 self._get_wordpieces('contains'),
                                                 self._get_wordpieces('contain'),
                                                 self._get_wordpieces('where'),
                                                 self._get_wordpieces('has'),]
        self._synonyms['predicates']['city'] = [self._get_wordpieces('cities'),
                                                self._get_wordpieces('city'),
                                                self._get_wordpieces('towns'),
                                                self._get_wordpieces('area'),]
        self._synonyms['predicates']['state'] = [self._get_wordpieces('states'),
                                                 self._get_wordpieces('state')]
        self._synonyms['predicates']['river'] = [self._get_wordpieces('rivers'),
                                                 self._get_wordpieces('river')]
        self._synonyms['predicates']['lake'] = [self._get_wordpieces('lakes'),]
        self._synonyms['predicates']['high_point_1'] = [self._get_wordpieces('high points'),
                                                        self._get_wordpieces('high point'),]
        self._synonyms['predicates']['next_to_2'] = [self._get_wordpieces('surrounding'),
                                                     self._get_wordpieces('surround'),
                                                     self._get_wordpieces('bordering'),
                                                     self._get_wordpieces('next to'),
                                                     self._get_wordpieces('border'),
                                                     self._get_wordpieces('neighboring'),
                                                     self._get_wordpieces('adjacent'),
                                                     self._get_wordpieces('borders'),
                                                     self._get_wordpieces('neighboring'),
                                                     self._get_wordpieces('neighbor'),
                                                     self._get_wordpieces('adjoin'),]
        self._synonyms['predicates']['next_to_1'] = [self._get_wordpieces('border'),
                                                     self._get_wordpieces('borders'),]
        self._synonyms['predicates']['traverse_2'] = [self._get_wordpieces('passes through'),
                                                      self._get_wordpieces('cross'),
                                                      self._get_wordpieces('traverse'),
                                                      self._get_wordpieces('run through'),
                                                      self._get_wordpieces('runs through'),
                                                      self._get_wordpieces('flowing through'),
                                                      self._get_wordpieces('flows through'),
                                                      self._get_wordpieces('flow though'),
                                                      self._get_wordpieces('flow through'),
                                                      self._get_wordpieces('goes through'),]
        self._synonyms['predicates']['traverse_1'] = [self._get_wordpieces('flow through'),
                                                      self._get_wordpieces('run through'),
                                                      self._get_wordpieces('washed'),
                                                      self._get_wordpieces('flow'),
                                                      self._get_wordpieces('run'),
                                                      self._get_wordpieces('runs'),
                                                      self._get_wordpieces('traversed'),
                                                      self._get_wordpieces('go through'),
                                                      self._get_wordpieces('running through'),
                                                      self._get_wordpieces('traverses'),
                                                      self._get_wordpieces('cross'),
                                                      self._get_wordpieces('pass through'),
                                                      self._get_wordpieces('lie on'),]
        self._synonyms['predicates']['capital'] = [self._get_wordpieces('capital'),
                                                   self._get_wordpieces('capitals')]
        self._synonyms['predicates']['highest'] = [self._get_wordpieces('highest'),
                                                   self._get_wordpieces('tallest'),
                                                   self._get_wordpieces('maximum'),]
        self._synonyms['predicates']['largest'] = [self._get_wordpieces('largest'),
                                                   self._get_wordpieces('biggest'),]
        self._synonyms['predicates']['longest'] = [self._get_wordpieces('longest'),
                                                   self._get_wordpieces('biggest'),
                                                   self._get_wordpieces('largest'),]
        self._synonyms['predicates']['shortest'] = [self._get_wordpieces('shortest'),]
        self._synonyms['predicates']['smallest'] = [self._get_wordpieces('smallest'),]
        self._synonyms['predicates']['fewest'] = [self._get_wordpieces('least'),
                                                  self._get_wordpieces('fewest'),]
        self._synonyms['predicates']['largest_one'] = [self._get_wordpieces('largest'),
                                                       self._get_wordpieces('greatest'),
                                                       self._get_wordpieces('highest'),
                                                       self._get_wordpieces('most'),
                                                       self._get_wordpieces('biggest'),]
        self._synonyms['predicates']['smallest_one'] = [self._get_wordpieces('least'),
                                                        self._get_wordpieces('smallest'),
                                                        self._get_wordpieces('lowest'),
                                                        self._get_wordpieces('sparsest')]
        self._synonyms['predicates']['lowest'] = [self._get_wordpieces('lowest'),]
        self._synonyms['predicates']['place'] = [self._get_wordpieces('point'),
                                                 self._get_wordpieces('points'),
                                                 self._get_wordpieces('elevation'),
                                                 self._get_wordpieces('spot'),]
        self._synonyms['predicates']['mountain'] = [self._get_wordpieces('mountain'),
                                                    self._get_wordpieces('mountains'),
                                                    self._get_wordpieces('peak'),]
        self._synonyms['predicates']['size'] = [self._get_wordpieces('big'),
                                                self._get_wordpieces('large'),
                                                self._get_wordpieces('size')]
        self._synonyms['predicates']['elevation_1'] = [self._get_wordpieces('how high'),
                                                       self._get_wordpieces('elevation'),
                                                       self._get_wordpieces('height'),
                                                       self._get_wordpieces('how tall'),]
        self._synonyms['predicates']['len'] = [self._get_wordpieces('long'),
                                               self._get_wordpieces('length'),]
        self._synonyms['predicates']['count'] = [self._get_wordpieces('how many'),
                                                 self._get_wordpieces('what is the number'),
                                                 self._get_wordpieces('number of')]
        self._synonyms['predicates']['major'] = [self._get_wordpieces('big'),
                                                 self._get_wordpieces('major')]
        self._synonyms['predicates']['population_1'] = [self._get_wordpieces('how many citizens'),
                                                        self._get_wordpieces('how many inhabitants'),
                                                        self._get_wordpieces('how many people'),
                                                        self._get_wordpieces('population'),
                                                        self._get_wordpieces('number of citizens'),
                                                        self._get_wordpieces('people'),
                                                        self._get_wordpieces('populations'),
                                                        self._get_wordpieces('populated'),
                                                        self._get_wordpieces('populous'),
                                                        self._get_wordpieces('how many residents'),
                                                        self._get_wordpieces('inhabitants'),]
        self._synonyms['predicates']['area_1'] = [self._get_wordpieces('how many square kilometers'),
                                                  self._get_wordpieces('area'),]
        self._synonyms['predicates']['density_1'] = [self._get_wordpieces('population density'),
                                                     self._get_wordpieces('average population'),
                                                     self._get_wordpieces('density'),
                                                     self._get_wordpieces('population densities'),
                                                     self._get_wordpieces('dense'),
                                                     self._get_wordpieces('average urban population'),]
        self._synonyms['predicates']['sum'] = [self._get_wordpieces('combined'),
                                               self._get_wordpieces('total'),]
        self._synonyms['predicates']['capital_1'] = [self._get_wordpieces('capital'),]
        self._synonyms['predicates']['capital_2'] = [self._get_wordpieces('capital'),]
        self._synonyms['predicates']['higher_2'] = [self._get_wordpieces('higher'),]
        self._synonyms['predicates']['lower_2'] = [self._get_wordpieces('lower'),]
        self._synonyms['predicates']['low_point_2'] = [self._get_wordpieces('elevations'),]
        self._synonyms['predicates']['most'] = [self._get_wordpieces('most')]
        self._synonyms['predicates']['exclude'] = [self._get_wordpieces('do not'),
                                                   self._get_wordpieces('which have no'),
                                                   self._get_wordpieces('does not'),
                                                   self._get_wordpieces('excluding'),
                                                   self._get_wordpieces('no'),
                                                   self._get_wordpieces('are not'),
                                                   self._get_wordpieces('not'),]
        self._synonyms['arguments']["'usa'"] = set([self._get_wordpieces('us'),
                                                    self._get_wordpieces('united states'),
                                                    self._get_wordpieces('the country'),
                                                    self._get_wordpieces('america'),])



        # filter_args = ['cube', 'sphere', 'cylinder', 'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow',
        #                'small', 'large', 'rubber', 'metal']
        # self._filter_args = [arg_instance for instances in [self._synonyms['arguments'][arg] for arg in filter_args] for arg_instance in instances]

    @overrides
    def _parse_action(self, string, location, tokens) -> Tree:
        predicate = tokens[0][0]
        if predicate == 'answer':
            return tokens[0][1]
        elif predicate == 'stateid' or predicate == 'countryid' or predicate == 'placeid' or predicate == 'riverid':
            arg = ' '.join(tokens[0][1:])
            self._synonyms['arguments'][arg].add(self._get_wordpieces(arg.replace("'",'')))
            hint_span = None
            # sub_tree = self._align_filter_to_text([arg], type='arguments', hint_span=hint_span)
            sub_tree = self._get_tree_from_constant(arg, type='arguments', constant_prefix=predicate)
            return sub_tree
        elif predicate == 'cityid':
            arg_1 = ' '.join(tokens[0][1:-1])
            arg_2 = tokens[0][-1]
            # assert arg_2 == '_'
            # todo: Handle cases like "cityid ( 'austin', 'tx' )"
            self._synonyms['arguments'][arg_1].add(self._get_wordpieces(arg_1.replace("'",'')))
            hint_span = None
            # sub_tree = self._align_filter_to_text([arg_1], type='arguments', hint_span=hint_span)
            sub_tree = self._get_tree_from_constant(arg_1, type='arguments', constant_prefix=predicate)
            return sub_tree
        elif predicate == 'loc_2' or predicate == 'city' or predicate == 'state' or predicate == 'river' \
                or predicate == 'next_to_2' or predicate == 'high_point_1' or predicate == 'capital'\
                or predicate == 'highest' or predicate == 'largest' or predicate == 'longest' or predicate == 'place'\
                or predicate == 'lake' or predicate == 'traverse_2' or predicate == 'size' or predicate == 'elevation_1'\
                or predicate == 'len' or predicate == 'shortest' or predicate == 'count' or predicate == 'major'\
                or predicate == 'population_1' or predicate == 'smallest' or predicate == 'largest_one'\
                or predicate == 'density_1' or predicate == 'loc_1' or predicate == 'area_1' or predicate == 'next_to_1'\
                or predicate == 'traverse_1' or predicate == 'lowest' or predicate == 'smallest_one' or predicate == 'sum'\
                or predicate == 'mountain' or predicate == 'capital_1' or predicate == 'fewest' or predicate == 'capital_2'\
                or predicate == 'higher_2' or predicate == 'lower_2' or predicate == 'low_point_2' \
                or predicate == 'most' or predicate == 'fewest':
            argument_tree = tokens[0][1]
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')
            if predicate not in pred_to_freq:
                pred_to_freq[predicate] = defaultdict(int)
            pred_to_freq[predicate][predicate_tree.get_node(predicate_tree.root).data.content] +=1

            if argument_tree == 'all':
                return predicate_tree
            else:
                arg_start, arg_end = (
                    int(argument_tree.root.split('-')[0]), int(argument_tree.root.split('-')[1]))
                pred_start, pred_end = (
                    int(predicate_tree.root.split('-')[0]), int(predicate_tree.root.split('-')[1]))
                if arg_start < pred_start < pred_end < arg_end:
                    num_children_arg_tree = len(argument_tree.children(argument_tree.root))
                    if num_children_arg_tree > 2:
                        raise ValueError('tree to make non-projective with has {} children'.format(num_children_arg_tree))
                    print('non-projective!!')
                    tree = self._combine_trees(predicate_tree, argument_tree)
                else:
                    tree = self._join_unary_predicate_tree(predicate_tree, argument_tree)
            return tree
        elif predicate == 'exclude':
            predicate_tree = self._get_tree_from_constant(predicate, 'predicates')
            if predicate not in pred_to_freq:
                pred_to_freq[predicate] = defaultdict(int)
            pred_to_freq[predicate][predicate_tree.get_node(predicate_tree.root).data.content] +=1

            tree = self._join_binary_predicate_tree(predicate=predicate_tree, arg_1=tokens[0][1], arg_2=tokens[0][2])
            return tree
        else:
            raise Exception('unknown_predicate={}'.format(predicate))

if __name__ == "__main__":

    span_mapper = SpanMapperGeo()
    tree_mapper = TreeMapperGeo()
    executor = ProgramExecutorGeo()

    # examples_path = ('datasets/geo/funql/dev.json', 'datasets/geo/funql/dev_spans.json')
    # examples_path = ('datasets/geo/funql/train.json', 'datasets/geo/funql/train_spans.json')
    # examples_path = ('datasets/geo/funql/test.json', 'datasets/geo/funql/test_spans.json')
    # examples_path = ('datasets/geo/funql/dev_template.json', 'datasets/geo/funql/dev_template_spans.json')
    # examples_path = ('datasets/geo/funql/train_template.json', 'datasets/geo/funql/train_template_spans.json')
    # examples_path = ('datasets/geo/funql/test_template.json', 'datasets/geo/funql/test_template_spans.json')

    # examples_path = ('datasets/geo/funql/dev_len.json', 'datasets/geo/funql/dev_len_spans.json')
    # examples_path = ('datasets/geo/funql/train_len.json', 'datasets/geo/funql/train_len_spans.json')
    examples_path = ('datasets/geo/funql/test_len.json', 'datasets/geo/funql/test_len_spans.json')


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

            # if 'intersection' in program:
            #     continue

            # if 'most' in program:
            #     continue

            # if i==136 or i==194 or i==267 or i==301 or i==348 or i==503 or i==504 or i==505 or i==525:
            #     continue

            if i == 77:
                print()

            # if i==1 or i==41 or i==191 or i==212 or i==258 or i==273:
            #     continue

            if i==125:
                print()


            if question == 'how many cities named austin are there in the usa ?':
                print('here')

            # parse_tree = span_mapper.map_prog_to_tree(question, program)
            # parse_tree.show(data_property="to_string")
            # is_valid = span_mapper.is_valid_tree(parse_tree)
            # if not is_valid:
            #     print('here')

            parses = []
            parse_tree = None
            try:
                parse_tree_1 = span_mapper.map_prog_to_tree(question, program)
                is_valid = span_mapper.is_valid_tree(parse_tree_1)
                is_projective = span_mapper.is_projective_tree(parse_tree_1)
                parses.append((parse_tree_1, is_valid, is_projective))
            except Exception as e:
                pass

            if (span_mapper._possible_chains or len(span_mapper._possible_constants) > 0):
                try:
                    parse_tree_2 = span_mapper.map_prog_to_tree(question, program)
                    is_valid = span_mapper.is_valid_tree(parse_tree_2   )
                    is_projective = span_mapper.is_projective_tree(parse_tree_2 )
                    parses.append((parse_tree_2, is_valid, is_projective))
                except Exception as e:
                    pass

            if len(parses) == 0:
                cannot_parse_trees.append((question, program))
                print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
                print('cannot_parse: {} {} \n {}'.format(i, question, traceback.print_exc()))
                print()
            else:
                if len(parses) >= 1:
                    parse_tree_curr = parses[0][0]
                    is_valid_curr = parses[0][1]
                    is_projective_curr = parses[0][2]
                    if not is_projective:
                        print()
                    if is_valid_curr and is_projective_curr:
                        parse_tree = parse_tree_curr
                        is_valid = is_valid_curr
                    elif len(parses) == 2:
                        parse_tree_curr_other = parses[1][0]
                        is_valid_curr_other = parses[1][1]
                        is_projective_curr_other = parses[1][2]
                        if not is_projective_curr_other:
                            print()
                        if is_valid_curr_other and is_projective_curr_other:
                            parse_tree = parse_tree_curr_other
                            is_valid = is_valid_curr_other
                        elif is_valid_curr:
                            parse_tree = parse_tree_curr
                            is_valid = is_valid_curr
                        elif is_valid_curr_other:
                            parse_tree = parse_tree_curr_other
                            is_valid_curr = is_valid_curr_other
                    elif len(parses) == 1:
                        if is_valid_curr:
                            parse_tree = parse_tree_curr
                            is_valid = is_valid_curr



            # should_try_parse = True
            # first_try = True
            # while should_try_parse:
            #     parse_tree = None
            #     try:
            #         parse_tree = span_mapper.map_prog_to_tree(question, program)
            #     # except ValueError as e:
            #     except Exception as e:
            #         # except ValueError as e:
            #         if first_try:
            #             first_try = False
            #             continue
            #         cannot_parse_trees.append((question, program))
            #         print('cannot_parse_trees={}'.format(len(cannot_parse_trees)))
            #         print('cannot_parse: {} {} \n {}'.format(i, question, traceback.print_exc()))
            #         print()
            #         break
            #
            #     is_valid = span_mapper.is_valid_tree(parse_tree)
            #     is_projective = span_mapper.is_projective_tree(parse_tree)
            #     if not is_valid:
            #         if first_try and (span_mapper._possible_chains or len(span_mapper._possible_constants) > 0):
            #             first_try = False
            #         elif first_try and not (
            #         (span_mapper._possible_chains or len(span_mapper._possible_constants) > 0)):
            #             print('non-projective!!!!')
            #             bad_unambiguous_trees.append((parse_tree, question, program))
            #             print('bad_trees={}'.format(len(bad_unambiguous_trees)))
            #             should_try_parse = False
            #             print()
            #         else:
            #             print('bad-ambig!!!!')
            #             bad_ambiguous_trees.append((parse_tree, question, program))
            #             print('bad_trees={}'.format(len(bad_ambiguous_trees)))
            #             should_try_parse = False
            #             print()
            #     else:
            #         should_try_parse = False

            if parse_tree and is_valid:
                good_parses.append((line, parse_tree))
                # program_bt = 'answer ( {} )'.format(tree_mapper.map_tree_to_program(parse_tree))
                program_bt = tree_mapper.map_tree_to_program(parse_tree)
                eq_bt = program == program_bt
                print(program)
                print(program_bt)
                print()
                if not eq_bt:
                    print()
                if not eq_bt:
                    denotation = executor.execute(program)
                    denotation_bt = executor.execute(program_bt)
                    if denotation != denotation_bt:
                        back_translation_fail.append((question, program, program_bt))
                # if not eq_bt and not program.startswith('query_attribute_equal') and not program.startswith(
                #         'count_equal'):
                #     raise Exception('Bad backtranslation failure={} \n {}'.format(question, program))
            # if parse_tree and not is_valid and not first_try:
            #     print('here')
            # if parse_tree and not is_valid:
            #     print('here')

    print('good_parses={}'.format(len(good_parses)))
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

    for ex in bad_unambiguous_trees:
        print('{} {}'.format(ex[1], ex[2]))

    # print(pred_to_freq)
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