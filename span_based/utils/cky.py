"""Searches for the top scoring span-tree using CKY."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Tuple, List, Set, Optional, Dict, Type

import numpy as np
import torch
from treelib import Tree

from span_based.utils.translate_programs_to_span_trees import Span
from span_based.utils.translate_span_trees_to_programs import TreeMapper
from span_based.utils.domain_utils import DomainUtils


NOT_SPAN_LABEL = 'NO-LABEL'
SPAN_LABEL = 'span'


class Grammar:
    """Binary grammar rules used for CKY."""

    def __init__(self, labels):
        self.labels = labels
        self.labels_reverse = {label: label_id for label_id, label in labels.items()}
        self._init_grammar()

    def get_not_span_nt(self):
        return NT.not_span

    def get_sentence_nt(self):
        return NT.sentence

    def _init_grammar(self):
        self.rules_binary = {
            NT.sentence: [(NT.kbc, NT.kbc), (NT.not_span, NT.kbc)],
            NT.kbc: [(NT.kbc, NT.kbc), (NT.kbc, NT.not_span),
                     (NT.not_span, NT.not_span)],
            NT.not_span: [(NT.not_span, NT.not_span)]
            }

        self.rules_unary = defaultdict(list)
        for label_id, label in self.labels.items():
            if label == NOT_SPAN_LABEL:
                self.rules_unary[NT.not_span].append(label_id)
            elif label != SPAN_LABEL:
                self.rules_unary[NT.kbc].append(label_id)

        self._init_relevant_indices_per_category()

    def _init_relevant_indices_per_category(self):
        """Saves the indices for all the categories of constants."""
        self.relevant_indices_per_category = {}
        relevant_indices = []
        relevant_categories = []
        for i, category in enumerate(self.labels.values()):
            if category != NOT_SPAN_LABEL and category != SPAN_LABEL:
                relevant_indices.append(i)
                relevant_categories.append(category)
        self.relevant_indices_per_category[NT.kbc] = (relevant_indices, np.array(relevant_categories))


class NT(Enum):
    """Non terminals."""
    sentence = 1
    kbc = 2
    not_span = 3


@dataclass
class Derivation:
    score: float
    category: str
    main_category: str
    span: Optional[Tuple[int, int]]
    left_bp: Type["Derivation"] = None  # back pointer to left child derivation
    right_bp: Type["Derivation"] = None  # back pointer to right child derivation
    middle_bp: Type["Derivation"] = None  # back pointer to middle child derivation (non-projective)
    split: Optional[int] = None  # Split point. No split for unary derivations
    can_join_left: Optional[bool] = True
    can_join_right: Optional[bool] = True


@dataclass
class SpanCell:
    """Represents a table cell for a certain span."""
    span: Tuple[int, int]
    predictions: List[float]
    scores: Dict[NT, List[Derivation]]
    # holds the best k constants for the cell, used for rules of the form
    # 'predicate -> not_span not_span'.
    max_constants: Dict[NT, Tuple[np.ndarray, np.ndarray]]


@dataclass
class CollectedSpan:
    """Represents a span that appears in the CKY predicted span tree."""
    category: str
    span: Tuple[int, int]


@dataclass
class MainCategory:
    """Represents the main category of a span, and children it was applied to."""
    category: str
    children: Set[str]


class CKY(object):

    def __init__(self, grammar: Grammar, tree_mapper: TreeMapper, domain_utils: DomainUtils):
        self._grammar = grammar
        self._tree_mapper = tree_mapper
        self._domain_utils = domain_utils

    def _get_span_cells(self, spans_prediction, spans, k):
        """Transforms prediction vector into a SpanCell instance."""

        span_cells = {}
        spans -= 1  # reduce 1 to ignore [cls] token
        spans = spans.tolist()

        spans_prediction = spans_prediction.numpy()

        for predictions, span in zip(spans_prediction, spans):
            start, end = span
            span_tup = (start, end)

            max_constants = {}
            if end-start > 1:  # relevant to spans larger than 1
                max_constants[NT.kbc] = self._get_max_constants(NT.kbc, predictions, k)

            span_cells[span_tup] = SpanCell(span=span_tup, predictions=predictions, scores={},
                                            max_constants=max_constants)
        return span_cells

    def _get_max_constants(self, constant, predictions, k):
        """Calculates top-k constants."""
        relevant_indices, relevant_categories = self._grammar.relevant_indices_per_category[constant]

        relevant_predictions = predictions[relevant_indices]
        if len(relevant_predictions) <= k:
            max_predictions = relevant_predictions
            max_categories = list(relevant_categories)
        else:
            max_predictions_indices = np.argpartition(-relevant_predictions, k)[:k]
            max_predictions = relevant_predictions[max_predictions_indices]
            max_categories = relevant_categories[max_predictions_indices]
        return max_categories, max_predictions

    def _follow_bps(self, derivation: Derivation, sentence: List[str], collected_spans: List[
        CollectedSpan]=None):
        """Transforms back-pointers collected in CKY into a span tree."""
        if not derivation.split:  # Stopping criteria for unary derivations.
            tree = Tree()
            start = derivation.span[0]
            end = derivation.span[1]
            identifier = '{}-{}'.format(start, end)
            span = Span(span=derivation.span, content=sentence[start: end], constant=derivation.category)
            tree.create_node(identifier=identifier, data=span)
            if collected_spans is not None:
                collected_spans.append(CollectedSpan(category=derivation.category,
                                                     span=(start + 1, end + 1)))
            return tree
        # follow left back pointer
        left_tree = self._follow_bps(derivation=derivation.left_bp, sentence=sentence,
                                     collected_spans=collected_spans)
        # follow right back pointer
        right_tree = self._follow_bps(derivation=derivation.right_bp, sentence=sentence,
                                      collected_spans=collected_spans)

        # merge children to tree
        top_tree = Tree()
        left_tree_span = left_tree.root.split('-')
        right_tree_span = right_tree.root.split('-')
        start = int(left_tree_span[0])
        end = int(right_tree_span[1])
        identifier = '{}-{}'.format(start, end)
        span = Span(span=(start, end), content=sentence[start: end], constant=derivation.category)
        top_tree.create_node(identifier=identifier, data=span)
        top_tree.paste(nid=identifier, new_tree=left_tree)
        if derivation.middle_bp is not None:
            middle_tree = self._follow_bps(derivation=derivation.middle_bp, sentence=sentence,
                                           collected_spans=collected_spans)
            top_tree.paste(nid=identifier, new_tree=middle_tree)
        top_tree.paste(nid=identifier, new_tree=right_tree)

        if collected_spans is not None:
            if derivation.category == SPAN_LABEL:
                # Make sure both spans are not some combination of not_span
                if not derivation.middle_bp or derivation.middle_bp.category != NOT_SPAN_LABEL :
                    # add 1 to match label spans
                    collected_spans.append(CollectedSpan(category=SPAN_LABEL, span=
                                                          (derivation.left_bp.span[0] + 1,
                                                         derivation.right_bp.span[1] + 1)))
            else:
                # add 1 to match label spans
                collected_spans.append(CollectedSpan(category=derivation.category,
                                                     span=(derivation.left_bp.span[0] + 1,
                                                           derivation.right_bp.span[1] + 1)))
        return top_tree

    def _initialize_unaries(self, span_cells, sent_len, k, sentence, no_semantics_words):
        """Initializes single tokens with their top-k predicted constants"""
        for i in range(sent_len):
            span_cell = span_cells[(i, i + 1)]
            for l_hand in self._grammar.rules_unary:
                # bad word-pieces
                word_piece = sentence[i]
                if word_piece in no_semantics_words and l_hand != NT.not_span:
                    span_cell.scores[l_hand] = []
                    continue
                r_hands = self._grammar.rules_unary[l_hand]
                scores = span_cell.predictions[r_hands]
                if len(scores) <= k:
                    max_derivations = r_hands
                    max_scores = scores
                else:
                    max_scores_indices = np.argpartition(-scores, k)[:k]
                    max_scores = scores[max_scores_indices]
                    max_derivations = np.array(r_hands)[max_scores_indices]
                categories = [self._grammar.labels[derivation] for derivation in max_derivations]
                derivations = [Derivation(score=score, category=category, main_category=MainCategory(category, set()),
                                          span=(i, i + 1))
                               for score, category in zip(max_scores, categories)]
                span_cell.scores[l_hand] = derivations

    def _determine_sub_root_scores(self, span_cells, l_child_cand, r_child_cand):
        """Gets the score for the JOIN category"""
        span = (l_child_cand.span[0], r_child_cand.span[1])
        sub_root_scores = [span_cells[span].predictions[self._grammar.labels_reverse[SPAN_LABEL]]]
        categories = [SPAN_LABEL]
        return sub_root_scores, categories

    def _determine_sub_root_scores_not_spans(self, span_cell, l_hand, not_span_nt, not_span_score):
        """Gets the score for terminal categories (constants and the empty category)"""
        if l_hand == not_span_nt:  # Corresponds to 'not_span -> not_span not_span'
            categories, sub_root_scores = ([NOT_SPAN_LABEL], [not_span_score])
        else:  # Corresponds to a KB constant made from multiple word-pieces
            categories, sub_root_scores = span_cell.max_constants[l_hand]
        return sub_root_scores, categories

    def _get_derivations_per_lhs(self, i, j, l_hand, span_cell, span_cells, k,
                                 not_span_nt, gold_program,
                                 allowed_combinations, consider_non_projective,
                                 sentence, no_semantics_words):
        """Calculates the top-k derivations for a specific left hand rule."""
        unique_scores = set()
        derivations: List[Derivation] = []
        not_span_score = span_cell.predictions[self._grammar.labels_reverse[NOT_SPAN_LABEL]]
        for l_non_terminal, r_non_terminal in self._grammar.rules_binary[l_hand]:
            # sub root scores for not_span not_span
            if l_non_terminal == not_span_nt and r_non_terminal == not_span_nt:
                sub_root_scores, categories = self._determine_sub_root_scores_not_spans(span_cell, l_hand, not_span_nt, not_span_score)
            else:
                sub_root_scores, categories = None, None

            # to have three children, all parts should not be not_span
            is_three_children_candidate = l_non_terminal == NT.kbc and r_non_terminal == NT.kbc

            # bad word-pieces
            if l_hand != not_span_nt and l_non_terminal == not_span_nt and r_non_terminal == not_span_nt and any(
                    [no_semantics_word in sentence[i:j] for no_semantics_word in no_semantics_words]):
                continue

            for s_1 in range(i + 1, j):
                self.calc_binary_rule(l_hand, categories, derivations, i, j,
                                      l_non_terminal, not_span_nt,
                                      r_non_terminal, s_1, span_cells, sub_root_scores,
                                      unique_scores, allowed_combinations)
                if consider_non_projective and is_three_children_candidate and j-i >= 3:
                    # In this case we consider 3 a derivation of 3 spans
                    self.calc_ternary_rule(derivations, i, j, l_non_terminal, not_span_nt,
                                           r_non_terminal, s_1, span_cells, unique_scores,
                                           gold_program, allowed_combinations)

        derivations.sort(key=lambda d: d.score, reverse=True)
        if l_hand != NT.sentence:
            max_derivations = derivations[:k]
        else:
            max_derivations = derivations
        return max_derivations

    def calc_main_category(self, l_category, r_category, allowed_combinations):
        if l_category.category == NOT_SPAN_LABEL:
            return MainCategory(r_category.category, set())
        elif r_category.category == NOT_SPAN_LABEL:
            return MainCategory(l_category.category, set())
        elif (l_category.category, r_category.category) in allowed_combinations:
            return MainCategory(l_category.category, set([r_category.category]))
        else:
            return MainCategory(r_category.category, set([l_category.category]))

    def calc_binary_rule(self, l_hand, categories, derivations, i, j, l_non_terminal,
                         not_span_nt, r_non_terminal, s, span_cells,
                         sub_root_scores, unique_scores, allowed_combinations):
        """Derives all derivations for a specific left hand rule and split point s."""

        l_child_cands = span_cells[(i, s)].scores[l_non_terminal]
        r_child_cands = span_cells[(s, j)].scores[r_non_terminal]
        for l_child_cand, r_child_cand in itertools.product(*[l_child_cands, r_child_cands]):
            can_join_right = True
            can_join_left = True
            if l_hand != not_span_nt and (l_non_terminal == not_span_nt or r_non_terminal == not_span_nt):
                if l_non_terminal != not_span_nt:
                    if not l_child_cand.can_join_right or l_child_cand.category == 'span':
                        continue
                    else:
                        can_join_right = False
                if r_non_terminal != not_span_nt:
                    if not r_child_cand.can_join_left:
                        continue
                    else:
                        can_join_left = False
            if l_non_terminal != not_span_nt or r_non_terminal != not_span_nt:
                sub_root_scores, categories = self._determine_sub_root_scores(
                    span_cells, l_child_cand, r_child_cand)
            child_score = l_child_cand.score + r_child_cand.score
            for sub_root_score, category in zip(sub_root_scores, categories):
                score = sub_root_score + child_score
                if score in unique_scores or score < -100000:
                    continue

                # During training, we prune compositions that do not appear in the gold program
                if allowed_combinations is not None:
                    l_main_category = l_child_cand.main_category
                    r_main_category = r_child_cand.main_category
                    # children are not in allowed combinations
                    if r_non_terminal != not_span_nt and l_non_terminal != not_span_nt and (
                            l_main_category.category, r_main_category.category) not in allowed_combinations and (
                                r_main_category.category, l_main_category.category) not in allowed_combinations:
                            continue
                    # children are in allowed combinations, but the predicate already has the argument as a child.
                    elif (l_main_category.category, r_main_category.category
                          ) in allowed_combinations and r_main_category.category in l_main_category.children and allowed_combinations.count(
                        (l_main_category.category, r_main_category.category)) == 1:
                        continue
                    elif (r_main_category.category, l_main_category.category
                          ) in allowed_combinations and l_main_category.category in r_main_category.children and allowed_combinations.count(
                        (r_main_category.category, l_main_category.category)) == 1:
                        continue
                    elif r_non_terminal == not_span_nt and l_non_terminal == not_span_nt:
                        main_category = MainCategory(category, set()) if category != SPAN_LABEL else MainCategory(NOT_SPAN_LABEL, set())
                    else:
                        main_category = self.calc_main_category(l_child_cand.main_category,
                                                 r_child_cand.main_category, allowed_combinations)
                else:
                    main_category = not_span_nt

                unique_scores.add(score)
                derivation = Derivation(score=score, category=category,
                                        main_category=main_category,
                                        span=(i, j), can_join_left=can_join_left, can_join_right=can_join_right,
                                        left_bp=l_child_cand, right_bp=r_child_cand, split=s)
                derivations.append(derivation)

    def calc_ternary_rule(self, derivations, i, j, l_non_terminal, not_span_nt,
                          r_non_terminal, s_1, span_cells, unique_scores, gold_program,
                          allowed_combinations):
        """Derives all derivations that have 3 children."""

        # iterate over the second split point
        for s_2 in range(s_1 + 1, j):
            m_child_cands = span_cells[(s_1, s_2)].scores[l_non_terminal]
            for l_non_terminal_outside, r_non_terminal_outside in self._grammar.rules_binary[r_non_terminal]:
                is_three_children_candidate_outside = l_non_terminal_outside != not_span_nt and r_non_terminal_outside != not_span_nt
                if not is_three_children_candidate_outside:
                    continue
                l_child_cands = span_cells[(i, s_1)].scores[l_non_terminal_outside]
                r_child_cands = span_cells[(s_2, j)].scores[r_non_terminal_outside]
                for l_child_cand, r_child_cand, m_child_cand in itertools.product(
                        *[l_child_cands, r_child_cands, m_child_cands]):
                    span = (l_child_cand.span[0], r_child_cand.span[1])
                    sub_root_score = span_cells[span].predictions[
                        self._grammar.labels_reverse[SPAN_LABEL]]
                    # The score for composing three spans together.
                    score = sub_root_score + l_child_cand.score + r_child_cand.score + m_child_cand.score
                    is_training = gold_program is not None
                    if score in unique_scores or (score < 0 if not is_training else score < -100000):
                        continue

                    # Enforce constraints during training
                    if allowed_combinations is not None:
                        if (m_child_cand.main_category.category, r_child_cand.main_category.category) in allowed_combinations:
                            # right outer child does not operate on left outer child
                            if (r_child_cand.main_category.category, l_child_cand.main_category.category) not in allowed_combinations:
                                continue
                            # middle child already operated on right child, or right operated on left
                            elif r_child_cand.main_category.category in m_child_cand.main_category.children or (
                                    l_child_cand.main_category.category in r_child_cand.main_category.children
                            ):
                                continue
                            main_category = MainCategory(m_child_cand.main_category.category, set(r_child_cand.main_category.category))
                        elif (m_child_cand.main_category.category, l_child_cand.main_category.category) in allowed_combinations:
                            if (l_child_cand.main_category.category, r_child_cand.main_category.category) not in \
                                    allowed_combinations:
                                continue
                            elif l_child_cand.main_category.category in m_child_cand.main_category.children or (
                                    r_child_cand.main_category.category in l_child_cand.main_category.children
                            ):
                                continue
                            main_category = MainCategory(m_child_cand.main_category.category,
                                                         set(l_child_cand.main_category.category))
                        else:
                            continue
                    else:
                        main_category = not_span_nt
                    unique_scores.add(score)
                    derivation = Derivation(score=score, category=SPAN_LABEL,
                                            main_category=main_category,
                                            span=(i, j),
                                            left_bp=l_child_cand, right_bp=r_child_cand,
                                            middle_bp=m_child_cand,
                                            split=(s_1, s_2))
                    derivations.append(derivation)

    def find_best_span_tree(self, sentence, spans_prediction, spans, k=5, gold_program=None):
        """Searches for the top scoring span tree for an utterance given model predictions."""
        sent_len = len(sentence)
        # Shift scores so empty category (not_span) score is zero.
        spans_prediction = spans_prediction - torch.unsqueeze(
            spans_prediction[:, self._grammar.labels_reverse[NOT_SPAN_LABEL]], dim=1)

        # During training we prune the search space using the gold program
        if gold_program:
            gold_program_parts, gold_program = self._domain_utils.get_constants(gold_program)
            is_index_irrelevant = []
            for i, category in enumerate(self._grammar.labels.values()):
                if not (category == NOT_SPAN_LABEL or category == SPAN_LABEL or category in
                        gold_program_parts):
                    is_index_irrelevant.append(i)
            # sets a low score for pruned categories
            spans_prediction[:, is_index_irrelevant] = -1000000
            allowed_combinations = self._domain_utils.get_allowed_program_combinations(gold_program)
        else:
            allowed_combinations = None

        span_cells = self._get_span_cells(spans_prediction, spans, k)
        not_span_nt = self._grammar.get_not_span_nt()

        if self._domain_utils.is_consider_no_semantics_words():
            no_semantics_words = ['is', 'are', 'the', 'it', 'which', 'do', 'does', 'a', '?', '.']
        else:
            no_semantics_words = []

        # Initialization
        self._initialize_unaries(span_cells, sent_len, k, sentence, no_semantics_words)

        # main pass over all spans
        for l in range(2, sent_len + 1):
            for i in range(sent_len):
                j = i+l
                if j > sent_len:
                    break
                span_cell = span_cells[(i, j)]
                for l_hand in self._grammar.rules_binary:
                    # No point in deriving sentence LHS for partial sentence, or not sentence LHS for full sentence.
                    if (l_hand == NT.sentence and (i != 0 or j != sent_len)) or (l_hand != NT.sentence and i == 0 and j == sent_len):
                        continue
                    # The best way to derive l_hand
                    span_cell.scores[l_hand] = self._get_derivations_per_lhs(
                        i, j, l_hand, span_cell, span_cells, k, not_span_nt, gold_program,
                        allowed_combinations,
                        consider_non_projective=self._domain_utils.is_consider_non_projective(),
                        sentence=sentence,
                        no_semantics_words=no_semantics_words)

        # Goes over all derivation for the full utterance in descending score order.
        for derivation in span_cells[(0, sent_len)].scores[self._grammar.get_sentence_nt()]:
            collected_spans = []
            span_tree = self._follow_bps(derivation=derivation, sentence=sentence,
                                         collected_spans=collected_spans)

            try:
                lf_pred = self._tree_mapper.map_tree_to_program(span_tree)
            except:
                lf_pred = None

            if lf_pred:
                if gold_program:
                    # During training, we return the tree if it is transformed to the gold program
                    if gold_program == lf_pred:
                        return span_tree, collected_spans
                else:
                    return span_tree, collected_spans
        return None, None
