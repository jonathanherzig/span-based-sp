"""Differenr utils which are domain spcific."""

from allennlp.common.registrable import Registrable
import pyparsing
import re
from typing import Tuple


class DomainUtils(Registrable):
    """Interface to be implemented for each domain."""

    def __init__(self, is_use_lexicon):
        self._is_use_lexicon = is_use_lexicon
        self._is_consider_no_semantics_words = True

    def is_entitiy(self, category) -> bool:
        """Checks whether the category is of an entity."""
        return NotImplementedError

    def get_allowed_program_combinations(self, program) -> Tuple[str, str]:
        """Gets pairs of constants (predicates or entities) which are allowed to compose."""
        return NotImplementedError

    def is_consider_non_projective(self):
        """Whether there are non-projective structures for programs in the domain."""
        return NotImplementedError

    def is_consider_no_semantics_words(self):
        return self._is_consider_no_semantics_words

    def _parse_action(self, string, location, tokens):
        """Parses a program bottom-up to get allowed compositions"""
        return NotImplementedError

    def get_constants(self, program):
        """Gets the predicates and entities in the program."""
        return NotImplementedError

    def get_lexicon_phrase(self, label):
        """Gets lexicon entries for a specific category."""
        lexicon_phrases = []
        if self.is_entitiy(label):
            lexicon_phrases.append(label.split('_')[-1].split('#')[-1].replace("'", ''))
        if self._is_use_lexicon and label in self._lexicon:
            lexicon_phrases.extend(self._lexicon[label])
        return lexicon_phrases


@DomainUtils.register("geo_domain_utils")
class DomainUtilsGeo(DomainUtils):

    def __init__(self, is_use_lexicon: bool = False, is_consider_no_semantics_words: bool = False):
        super().__init__(is_use_lexicon)
        self._parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self._parser.setParseAction(self._parse_action)

        self._lexicon = {
            'state': ['state', 'states'],
            'river': ['river', 'rivers'],
            'city': ['city', 'cities'],
            'mountain': ['mountain', 'peak'],
            'lake': ['lake', 'lakes'],
            'capital': ['capital', 'capitals'],
            'capital_1': ['capital', 'capitals'],
            'capital_2': ['capital', 'capitals'],
            'loc_2': ['in', 'of'],
            'loc_1': ['in', 'with'],
            'count': ['how many', 'what is the number'],
            'high_point_1': ['high points', 'high point'],
            'next_to_2': ['border', 'borders'],
            'next_to_1': ['border', 'borders'],
            'traverse_2': ['flow through', 'run through'],
            'traverse_1': ['runs', 'run through'],
            'highest': ['highest', 'tallest'],
            'largest': ['largest', 'biggest'],
            'longest': ['longest', 'biggest'],
            'shortest': ['shortest', 'smallest'],
            'smallest': ['smallest'],
            'fewest': ['least', 'fewest'],
            'largest_one': ['largest', 'most'],
            'smallest_one': ['least', 'smallest'],
            'lowest': ['lowest'],
            'place': ['point', 'elevation'],
            'size': ['size', 'big'],
            'elevation_1': ['how high', 'elevation'],
            'len': ['long', 'length'],
            'major': ['big', 'major'],
            'population_1': ['how many people', 'population'],
            'area_1': ['area', 'how many square kilometers'],
            'density_1': ['population density', 'density'],
            'sum': ['combined', 'total'],
            'higher_2': ['higher'],
            'lower_2': ['lower'],
            'low_point_2': ['elevations'],
            'most': ['most'],
            'exclude': ['excluding', 'no'],
            "countryid#\'usa\'": ['us', 'united states']
        }

    def is_entitiy(self, category):
        return '#' in category

    def is_consider_non_projective(self):
        return True

    def get_allowed_program_combinations(self, program):
        self.allowed_combinations = []
        program = re.sub(r'(\w+) \(', r'( \1', program)
        program = program.replace(',', '')
        self._parser.parseString(program)[0]
        return self.allowed_combinations

    def _parse_action(self, string, location, tokens):
        predicate = tokens[0][0]
        argument = tokens[0][1]
        if predicate == 'cityid':
            argument = ' '.join(tokens[0][1:-1])
            return predicate+'#'+argument
        elif predicate == 'stateid' or predicate == 'countryid' or predicate == 'placeid' or predicate == 'riverid':
            argument = ' '.join(tokens[0][1:])
            return predicate + '#' + argument
        elif predicate == 'exclude':
            self.allowed_combinations.append((predicate, tokens[0][1]))
            self.allowed_combinations.append((predicate, tokens[0][2]))
            return predicate
        elif predicate == 'answer':
            return
        elif argument == 'all':
            return predicate
        else:
            self.allowed_combinations.append((predicate, argument))
            return predicate

    def get_constants(self, program):
        program = re.sub("cityid \( (.+?), ['_a-z]+ \)", r'cityid ( \1, _ )', program)
        program_for_parts = re.sub("stateid \( ('.+?') \)", r'stateid#\1', program)
        program_for_parts = re.sub("countryid \( ('.+?') \)", r'countryid#\1', program_for_parts)
        program_for_parts = re.sub("placeid \( ('.+?') \)", r'placeid#\1', program_for_parts)
        program_for_parts = re.sub("riverid \( ('.+?') \)", r'riverid#\1', program_for_parts)
        program_for_parts = re.sub("cityid \( ('.+?')", r'cityid#\1', program_for_parts)
        program_for_parts = re.sub("'(\w+) (\w+)'", "'" + r"\1*\2" + "'", program_for_parts)
        program_for_parts = re.sub("'(\w+) (\w+) (\w+)'", "'" + r"\1*\2*\3" + "'", program_for_parts)

        constants = program_for_parts.replace('(', '').replace(')', '').replace(',', '').split()
        constants = [constant.replace('*', ' ') for constant in constants]
        # Remove some fragments which are not KB constants.
        for fragment in ['answer', 'all', '_']:
            if fragment in constants:
                constants.remove(fragment)
        return constants, program


@DomainUtils.register("clevr_domain_utils")
class DomainUtilsClevr(DomainUtils):

    def __init__(self, is_use_lexicon: bool = False, is_consider_no_semantics_words: bool = False):
        super().__init__(is_use_lexicon)
        self._parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self._parser.setParseAction(self._parse_action)

        self._lexicon = {
            'large': ['big'],
            'metal': ['shiny', 'metallic'],
            'cube': ['block', 'blocks'],
            'count_greater': ['greater than', 'more'],
            'small': ['tiny'],
            'rubber': ['matte'],
            'relate_attribute_equal': ['same'],
            'count': ['how many', 'what number'],
            'sphere': ['ball', 'spheres'],
            'query_attribute_equal': ['same'],
            'cylinder': ['cylinders'],
            'intersect': ['and'],
            'query': ['what', 'how'],
            'material': ['made of'],
            'size': ['size', 'big'],
            'exist': ['there', 'any'],
            'union': ['or'],
            'count_equal': ['equal number', 'same number'],
            'count_less': ['fewer', 'less'],
        }

    def is_entitiy(self, category):
        entities = ['cube', 'sphere', 'cylinder', 'gray', 'red', 'blue', 'green', 'brown', 'purple',
                    'cyan', 'yellow', 'small', 'large', 'rubber', 'metal', 'shape', 'color',
                    'material', 'size', 'left', 'right', 'behind', 'front']
        return category in entities

    def is_consider_non_projective(self):
        return False

    def get_allowed_program_combinations(self, program):
        self.allowed_combinations = []
        program = re.sub(r'(\w+) \(', r'( \1', program)
        program = program.replace(',', '')
        self._parser.parseString(program)[0]
        return self.allowed_combinations

    def _parse_action(self, string, location, tokens):
        predicate = tokens[0][0]
        if predicate == 'scene':
            return predicate
        elif predicate == 'filter':
            main_arg = tokens[0][1]
            for i in range(2, len(tokens[0])):
                arg = tokens[0][i]
                if arg != 'scene':
                    self.allowed_combinations.append((main_arg, arg))
            return main_arg
        elif predicate == 'relate':
            main_arg = tokens[0][1]
            self.allowed_combinations.append((main_arg, tokens[0][2]))
            return main_arg
        else:
            main_arg = tokens[0][0]
            for i in range(1, len(tokens[0])):
                arg = tokens[0][i]
                self.allowed_combinations.append((main_arg, arg))
            return main_arg

    def get_constants(self, program):
        constants = program.replace('(', '').replace(')', '').replace(',', '').split()
        # Remove some fragments which are not KB constants.
        constants = [c for c in constants if c not in ['filter', 'scene', 'relate']]
        return constants, program


@DomainUtils.register("scan_domain_utils")
class DomainUtilsScan(DomainUtils):

    def __init__(self, is_use_lexicon: bool = False, is_consider_no_semantics_words: bool = False):
        super().__init__(is_use_lexicon)
        self._parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self._parser.setParseAction(self._parse_action)

        self._lexicon = {
            'i_jump': ['jump'],
            'i_run': ['run'],
            'i_look': ['look'],
            'i_turn': ['turn'],
            'i_walk': ['walk'],
            'i_and': ['and'],
            'i_after': ['after'],
            'i_twice': ['twice'],
            'i_thrice': ['thrice'],
        }

    def is_entitiy(self, category):
        entities = ['i_right', 'i_left', 'i_around', 'i_opposite']
        return category in entities

    def is_consider_non_projective(self):
        return False

    def get_allowed_program_combinations(self, program):
        self.allowed_combinations = []
        program = re.sub(r'(\w+) \(', r'( \1', program)
        program = program.replace(',', '')
        self._parser.parseString(program)[0]
        return self.allowed_combinations

    def _parse_action(self, string, location, tokens):
        predicate = tokens[0][0]
        for i in range(1, len(tokens[0])):
            arg = tokens[0][i]
            self.allowed_combinations.append((predicate, arg))
        return predicate

    def get_constants(self, program):
        return program.replace('(', '').replace(')', '').replace(',', '').split(), program
