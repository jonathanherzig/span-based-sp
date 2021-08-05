from itertools import product
import json
import re
from typing import List, Tuple

from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import BertPreTokenizer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
import pyparsing
from treelib import Tree


class Span(object):

    def __init__(self, span: Tuple[int, int], content: List[str], constant: str, height: int = None):
        self.height = height
        self.span = span
        self.content = content
        self.constant = constant
        self.to_string = '{} - {} - "{}"'.format(span, constant, ' '.join(content))


class SpanMapper(object):

    def __init__(self):
        self._tokenizer = BertPreTokenizer() # should align with reader's tokenizer
        self._token_indexers = PretrainedBertIndexer(pretrained_model='bert-base-uncased') # should align with reader's tokenizer
        self._synonyms = {'arguments': {},
                          'predicates': {}}
        # self._load_argumets()
        # self._enrich_synonyms()
        # self._enrich_synonyms_by_hand()
        self._parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        # self._parser.setParseAction(self._parse_action)

    def _parse_action(self, string, location, tokens) -> Tree:
        raise NotImplementedError

    def _get_wordpieces(self, text): # should match the reader method
        tokens = self._tokenizer.tokenize(text)
        do_lowercase = True
        tokens_out = (
            token.text.lower()
            if do_lowercase and token.text not in self._tokenizer.never_split
            else token.text
            for token in tokens
        )
        wps = [
            [wordpiece for wordpiece in self._token_indexers.wordpiece_tokenizer(token)]
            for token in tokens_out
        ]
        wps_flat = [wordpiece for token in wps for wordpiece in token]
        return tuple(wps_flat)

    def _align_to_text(self, constant, type):
        spans = []
        realizations = self._synonyms[type][constant]
        num_tokens_text = len(self._tokens)
        for realization in realizations:
            num_tokens_real = len(realization)
            for begin in range(num_tokens_text - num_tokens_real + 1):
                end = begin + num_tokens_real
                if self._tokens[begin : end] == realization:
                    span = Span(height=1, span=(begin, end), content=self._tokens[begin : end], constant=constant)
                    spans.append(span)
        # assert len(spans) == 1
        return spans

    def _align_filter_to_text(self, constants, type, hint_span=None):
        spans_per_constant = []
        for i, constant in enumerate(constants):
            constant_spans = self._align_to_text(constant, type)  # find spans in text for a particular constant
            # if len(constants) == 1 and hint_span != None:
            # # if i == 0 and hint_span != None:
            #     constant_spans_ = self._choose_span_by_hint(constant_spans, hint_span)
            #     if constant_spans_ != constant_spans:
            #         print('here')
            #         constant_spans = constant_spans_
            # if len(constants) == 2 and hint_span != None and i == 1 and len(constant_spans) > 1:
            #     constant_spans__ = []
            #     for j, span in enumerate(constant_spans):
            #         # temporary bad way to disambiguate spans
            #         if span.span[0] < 4 and hint_span.span[0] <= 6:
            #             constant_spans__.append(span)
            #     if len(constant_spans__) > 0:
            #         print('here') # risky
            #         constant_spans = constant_spans__
            # assert len(constant_spans) >= 1
            spans_per_constant.append(constant_spans)
        contiguous_chains = self._find_contiguous_chain(spans_per_constant)
        # if len(contiguous_chains) > 1 and hint_span:
        #     contiguous_chains_ = self._filter_chains_with_hint(contiguous_chains, hint_span)
        #     if contiguous_chains_ != contiguous_chains:
        #         print('here') # risky
        #     contiguous_chains = contiguous_chains_
        if len(contiguous_chains) == 2:
            if not self._possible_chains:
                next_option = tuple([span.constant for span in contiguous_chains[1]])
                self._possible_chains = contiguous_chains[1]
                contiguous_chains = [contiguous_chains[0]]
            else:
                contiguous_chains = [self._possible_chains]
                self._possible_chains = None
        if len(contiguous_chains) != 1:
            print('here')
        assert len(contiguous_chains) == 1
        contiguous_chain = contiguous_chains[0]
        return self._contiguous_chain_to_subtree(contiguous_chain)

    def _filter_chains_with_hint(self, contiguous_chains, hint_span):
        # min delta from span edges. Sets a high values if span is within the hint_span
        delta_from_hint = [min(abs(chain[0].span[0] - hint_span.span[1]), abs(chain[-1].span[1] - hint_span.span[0])
                               ) if not self._is_span_contained(chain[0], hint_span) else 10000 for chain in contiguous_chains]
        min_value = min(delta_from_hint)
        min_chains = []
        for i, chain in enumerate(contiguous_chains):
            if delta_from_hint[i] == min_value:
                min_chains.append(chain)
        # if min_value == 0 and len(min_spans) > 1:  # take the last span if all min_values are 0
        #     min_spans = [sorted(spans, key=lambda span: span.span[0], reverse=True)[0]]
        return min_chains

    def _filter_sub_chains(self, contiguous_chains):
        """Filter chains that are actually a part of a larger chain"""
        full_contiguous_chains = []
        tokens = self._tokens
        for chain in contiguous_chains:
            start = chain[0].span[0]
            end = chain[-1].span[1]
            if start > 0:  # there is at least one token before 'start'
                if tuple(tokens[start-1: start]) in self._filter_args:
                    continue
            if start > 1:  # there are at least two tokens before 'start'
                if tuple(tokens[start-2: start]) in self._filter_args:
                    continue
            if tuple(self._tokens[end: end + 1]) in self._filter_args:
                continue
            if tuple(self._tokens[end: end + 2]) in self._filter_args:
                continue
            full_contiguous_chains.append(chain)
        return full_contiguous_chains

    def _find_contiguous_chain(self, spans_per_constant: List[List[Span]]):
        contiguous_chains = []
        combinations = list(product(*spans_per_constant))
        for comb in combinations:
            if all([s.span[1] == comb[i+1].span[0] for i, s in enumerate(comb[:-1])]): # check if contiguous
                if not any([self._is_span_contained(sub_span, span) for sub_span in comb for span in self._decided_spans]):  # check if span wasn't decided before
                    contiguous_chains.append(comb)
                else:
                    print('here')
        if len(contiguous_chains) > 1:
            contiguous_chains_ = self._filter_sub_chains(contiguous_chains)
            if contiguous_chains != contiguous_chains_:
                print('here')
            contiguous_chains = contiguous_chains_
        return contiguous_chains

    def _contiguous_chain_to_subtree(self, contiguous_chain: List[Span]):
        self._decided_spans += contiguous_chain
        tree = Tree()
        stack = []
        for i in range(len(contiguous_chain)-1):
            # make parent
            start = contiguous_chain[i].span[0]
            end = contiguous_chain[-1].span[1]
            span = Span(height=len(contiguous_chain)-1, span=(start, end), content=self._tokens[start:end], constant=None)
            parent = stack[-1] if len(stack) > 0 else None
            identifier = '{}-{}'.format(start, end)
            tree.create_node(identifier=identifier, data=span, parent=parent)
            stack.append(identifier)

            # make left child
            span_lc = contiguous_chain[i]
            identifier_lc = '{}-{}'.format(span_lc.span[0], span_lc.span[1])
            tree.create_node(identifier=identifier_lc, data=span_lc, parent=identifier)

        # make last right child
        span_rc = contiguous_chain[-1]
        identifier_rc = '{}-{}'.format(span_rc.span[0], span_rc.span[1])
        tree.create_node(identifier=identifier_rc, data=span_rc, parent=stack[-1] if stack else None)

        return tree  # return top most identifier

    def _join_trees(self, subtree_1: Tree, subtree_2: Tree):

        top_tree = Tree()
        arg_1_span = subtree_1.root.split('-')
        arg_2_span = subtree_2.root.split('-')
        start = int(arg_1_span[0])
        end = int(arg_2_span[1])
        identifier = '{}-{}'.format(start, end)
        span = Span(height=100, span=(start, end), content=self._tokens[start:end], constant=None)
        top_tree.create_node(identifier=identifier, data=span)
        top_tree.paste(nid=identifier, new_tree=subtree_1)
        top_tree.paste(nid=identifier, new_tree=subtree_2)
        return top_tree

    def _combine_trees(self, subtree_1: Tree, subtree_2: Tree):
        subtree_2.paste(nid=subtree_2.root, new_tree=subtree_1)
        return subtree_2

    def _join_binary_predicate_tree(self, predicate: Tree, arg_1: Tree, arg_2: Tree, allow_arg_switch: bool = True):
        predicate_start = int(predicate.root.split('-')[0])
        arg_1_start = int(arg_1.root.split('-')[0])
        arg_2_start = int(arg_2.root.split('-')[0])

        # if arg_2 is not in the middle
        if not (arg_2_start > predicate_start and arg_2_start < arg_1_start) and not (arg_2_start < predicate_start and arg_2_start > arg_1_start):
            if predicate_start < arg_1_start:  # predicate is left to arg_1
                join_1 = self._join_trees(predicate, arg_1)
            else:
                join_1 = self._join_trees(arg_1, predicate)

            if predicate_start < arg_2_start:  # predicate is left to arg_2
                join_2 = self._join_trees(join_1, arg_2)
            else:
                join_2 = self._join_trees(arg_2, join_1)
            return join_2
        else:
            if not allow_arg_switch:  # in this case we do not allow arg_2 to be in a span with the predicate
                raise Exception('Argument switch is not allowed={}'.format(predicate))
            if predicate_start < arg_1_start:  # predicate is left to arg_1, and arg_2 is in the middle
                join_1 = self._join_trees(predicate, arg_2)
                join_2 = self._join_trees(join_1, arg_1)
            else:
                join_1 = self._join_trees(arg_2, predicate)
                join_2 = self._join_trees(arg_1, join_1)
            return join_2

    def _join_unary_predicate_tree(self, predicate: Tree, arg: Tree):
        predicate_start = int(predicate.root.split('-')[0])
        arg_start = int(arg.root.split('-')[0])


        if predicate_start < arg_start:  # predicate is left to arg_1
            join_tree = self._join_trees(predicate, arg)
        else:
            join_tree = self._join_trees(arg, predicate)

        return join_tree

    def _filter_contained_spans(self, spans):
        spans_ = sorted(spans, key=lambda span: span.span[1] - span.span[0], reverse=True)  # sort according to span length
        if spans_ != spans:
            print('here')
        spans = spans_
        filtered_spans = []
        for span in spans:
            for broad_span in filtered_spans:
                if self._is_span_contained(span, broad_span): # span contained in broad_span
                    break
            else:
                filtered_spans.append(span)
        return filtered_spans

    def _is_span_contained(self, span_1: Span, span_2: Span):
        return set(range(span_1.span[0], span_1.span[1])).issubset(set(range(span_2.span[0], span_2.span[1])))

    def _is_spans_intersect(self, span_1: Span, span_2: Span):
        return len(set(range(span_1.span[0], span_1.span[1])).intersection(set(range(span_2.span[0], span_2.span[1])))) > 0

    def _choose_span_by_hint(self, spans, hint_span):
        """Chooses that span that is closest to the hint span. The hint span is the one the selected span should be close to."""

        # min delta from span edges. Sets a high values if span is within the hint_span
        delta_from_hint = [min(abs(span.span[0] - hint_span.span[1]), abs(span.span[1] - hint_span.span[0])
                               ) if not self._is_span_contained(span, hint_span) else 10000 for span in spans]
        min_value = min(delta_from_hint)
        min_spans = []
        for i, span in enumerate(spans):
            if delta_from_hint[i] == min_value:
                min_spans.append(span)
        if min_value == 0 and len(min_spans) > 1:  # take the last span if all min_values are 0
            min_spans = [sorted(spans, key=lambda span: span.span[0], reverse=True)[0]] # risky
        return min_spans

    def _get_tree_from_constant(self, constant, type, hint_span=None, constant_prefix=None):
        spans = self._align_to_text(constant, type)
        spans_ = self._filter_contained_spans(spans)
        if spans != spans_:
            print('here')
        spans = spans_
        if len(spans) != 1:
            print('here')
        # if len(spans) > 1 and hint_span:
        #     spans__ = self._choose_span_by_hint(spans, hint_span)
        #     if spans != spans__:
        #         print('here')
        #     spans = spans__
        spans___ = []
        for span in spans:
            if not any([self._is_span_contained(span, decided_span) for decided_span in self._decided_spans]):
                spans___.append(span)
        if spans___ != spans:
            print('here')
        spans = spans___
        if len(spans) == 2:
            if not constant in self._possible_constants:
                self._possible_constants[constant] = spans[1]
                spans = [spans[0]]
            else:
                spans = [self._possible_constants[constant]]
                del self._possible_constants[constant]
        if len(spans) != 1:
            print('spans for {} are {}'.format(constant, spans))
        assert len(spans) == 1
        span = spans[0]
        if constant_prefix:
            span.constant = '{}#{}'.format(constant_prefix, span.constant)
        self._decided_spans.append(span)
        identifier = '{}-{}'.format(span.span[0], span.span[1])
        constant_tree = Tree()
        constant_tree.create_node(identifier=identifier, data=span)
        return constant_tree

    def is_valid_tree(self, parse_tree: Tree):
        is_violateing = [self._is_violating_node(node, parse_tree) for node in parse_tree.expand_tree()]
        if any(is_violateing):
            print('here')
        return not any(is_violateing)

    def is_projective_tree(self, parse_tree: Tree):
        is_violateing = [len(parse_tree.children(node)) > 2 for node in parse_tree.expand_tree()]
        if any(is_violateing):
            print('here')
        return not any(is_violateing)

    def _is_violating_node(self, node, parse_tree):
        """Checks id a node is violated - if its child's span is not contained in its parent span, or intersect another child."""
        node_span = parse_tree.get_node(node).data
        for child in parse_tree.children(node):
            child_span = child.data
            if not self._is_span_contained(child_span, node_span):  # not contained in parent's span
                print('node {} is not contained in parent {}'.format(child_span.to_string, node_span.to_string))
                return True
            for child_other in parse_tree.children(node):
                child_other_span = child_other.data
                if child != child_other and self._is_spans_intersect(child_span, child_other_span):  # intersects another span
                    print('node {} intersectes node {}'.format(child_span.to_string, child_other_span.to_string))
                    return True
        return False

    def map_prog_to_tree(self, question, program):
        program = re.sub(r'(\w+) \(', r'( \1', program)
        self._program = program.replace(',', '')
        self._tokens = self._get_wordpieces(question)
        self._tree = Tree()
        self._decided_spans = []
        parse_result = self._parser.parseString(self._program)[0]
        return parse_result



        # # print the program tree
        # executor.parser.setParseAction(_parse_action_tree)
        # tree_parse = executor.parser.parseString(program)[0]
        # print('parse_tree=')
        # pprint(tree_parse)

    # def pprint(node, tab=""):
    #     if isinstance(node, str):
    #         print(tab + u"┗━ " + str(node))
    #         return
    #     print(tab + u"┗━ " + str(node.value))
    #     for child in node.children:
    #         pprint(child, tab + "    ")

    def _parse_action_tree(string, location, tokens):
        from collections import namedtuple
        Node = namedtuple("Node", ["value", "children"])
        node = Node(value=tokens[0][0], children=tokens[0][1:])
        return node

    def _get_aligned_span(self, subtree: Tree):
        """Gets hint span for aligning ambiguous constants (e.g., 'left' appears twice)"""
        return subtree.get_node(subtree.root).data

    def _get_first_argument_to_join(self, predicate_tree, arg1_tree, arg2_tree):
        predicate_span_start = int(predicate_tree.root.split('-')[0])
        arg1_span_start = int(arg1_tree.root.split('-')[0])
        arg2_span_start = int(arg2_tree.root.split('-')[0])
        if predicate_span_start < arg2_span_start < arg1_span_start or predicate_span_start > arg2_span_start > arg1_span_start:
            return arg2_tree, arg1_tree
        else:
            return arg1_tree, arg2_tree


    def _get_details(self, child, span_labels):
        data = child.data
        start = data.span[0]
        end = data.span[1] - 1
        span = (data.span[0], data.span[1] - 1)
        type = data.constant if data.constant else 'span'
        is_span = type == 'span'
        if not is_span:
            span_labels.append({'span': span, 'type': type})
        return start, end, is_span

    def _adjust_end(self, start, end, adjusted_end, is_span, span_labels):
        if end < adjusted_end-1:
            span_labels.append({'span': (start, adjusted_end-1), 'type': 'span'})
        else:
            if is_span:
                span_labels.append({'span': (start, end), 'type': 'span'})

    def _inner_write(self, span_labels, children, end, parse_tree):
        children.sort(key=lambda c: c.data.span[0])
        start_1, end_1, is_span_1 = self._get_details(children[0], span_labels)
        start_2, end_2, is_span_2= self._get_details(children[1], span_labels)
        if len(children) > 2:
            start_3, end_3, is_span_3 = self._get_details(children[2], span_labels)

        self._adjust_end(start_1, end_1, start_2, is_span_1, span_labels)

        if len(children) > 2:
            self._adjust_end(start_2, end_2, start_3, is_span_2, span_labels)
            self._adjust_end(start_3, end_3, end+1, is_span_3, span_labels)
            # if end_2 < start_3 - 1:
            #     span_labels.append({'span': (start_2, start_3 - 1), 'type': 'span'})
            # if end_3 < end:
            #     span_labels.append({'span': (start_3, end), 'type': 'span'})
        else:
            self._adjust_end(start_2, end_2, end+1, is_span_2, span_labels)
            # if end_2 < end:
            #     span_labels.append({'span': (start_2, end), 'type': 'span'})

        children_1 = parse_tree.children(children[0].identifier)
        if len(children_1) > 0:
            self._inner_write(span_labels, children_1, start_2-1, parse_tree)
        children_2 = parse_tree.children(children[1].identifier)
        if len(children) > 2:
            children_3 = parse_tree.children(children[1].identifier)
            if len(children_2) > 0:
                self._inner_write(span_labels, children_2, start_3-1, parse_tree)
            if len(children_3) > 0:
                self._inner_write(span_labels, children_3, end, parse_tree)
        else:
            if len(children_2) > 0:
                self._inner_write(span_labels, children_2, end, parse_tree)

    def write_to_output(self, line, parse_tree, output_file):
        tokens = self._get_wordpieces(line['question'])

        if line['question'] == "what state borders michigan ?":
            print()
        len_sent = len(tokens)

        span_labels = []

        # type = 'span'
        # span_labels.append({'span': (0, len_sent-1), 'type': type})
        root = parse_tree.root
        root_node = parse_tree.get_node(root).data

        # root_start, root_end = root.split('-')
        # root_start = int(root_start)
        # root_end = int(root_end)
        #
        # if root_start > 0:
        #
        start = 0
        end = len_sent - 1
        type = 'span'
        span_labels.append({'span': (start, end), 'type': type})

        children = parse_tree.children(root)


        if len(children) == 0:
            s, t = (int(root_node.span[0]), int(root_node.span[1]))
            type = root_node.constant
            span_labels.append({'span': (s,t), 'type': type})
            if s > 0:
                span_labels.append({'span': (s, end), 'type': 'span'})
        else:
            child_1_start = children[0].data.span[0]
            if child_1_start > 0:
                span_labels.append({'span': (child_1_start, end), 'type': 'span'})
            self._inner_write(span_labels, parse_tree.children(root), end, parse_tree)

        # while (len(parse_tree.children(root)) > 0):
        #     children = parse_tree.children(root)
        #     data_1 = children[0].data
        #     span_1 = (data_1.span[0], data_1.span[1] - 1)
        #     data_2 = children[1].data
        #     span_2 = (data_2.span[0], data_2.span[1] - 1)
        #     print()


        # for i, node in enumerate(parse_tree.expand_tree()):
        #     data = parse_tree.get_node(node).data
        #     span = (data.span[0], data.span[1]-1)  # move to inclusive spans
        #     if i==0:
        #         left_extra = None
        #         if span[0] > 0:
        #             left_extra = (0, span[0]-1)
        #         right_extra = None
        #         if span[1] < len_sent-1:
        #             left_right = (span[1]+1, len_sent-1)
        #
        #     type = data.constant if data.constant else 'span'
        #     span_labels.append({'span': span, 'type': type})
        line['gold_spans'] = span_labels
        json_str = json.dumps(line)
        output_file.write(json_str + '\n')