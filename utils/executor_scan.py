"""An executor for SCAN-SP programs."""

import json
from overrides import overrides
import time
import re

import pyparsing

from utils.executor import Executor

PREFIX = 'i_'
SUFFIX = ''


@Executor.register("scan_executor")
class ProgramExecutorScan(Executor):

    def __init__(self):
        self.parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self.parser.setParseAction(self._parse_action)

    @overrides
    def execute(self, program: str, kb_str: str = None) -> str:

        program = re.sub(r'(\w+) \(', r'( \1', program)  # transform to prefix notation
        program = program.replace(',', '')
        self._scene_str = kb_str

        try:
            parse = self.parser.parseString(program)
            parse = parse[0]
        except Exception as e:
            return 'error_parse: {}'.format(e)

        if not isinstance(parse, str):
            return 'error_type: parse is not string, but {}'.format(type(parse))

        return parse

    def _parse_action(self, string, location, tokens):
        predicate = tokens[0][0]
        length = len(tokens[0])
        if predicate in [PREFIX+'jump'+SUFFIX, PREFIX+'run'+SUFFIX,
                         PREFIX+'look'+SUFFIX, PREFIX+'turn'+SUFFIX,
                         PREFIX+'walk'+SUFFIX]:
            predicate_str = 'I_{}'.format(predicate.replace(PREFIX, '').replace(SUFFIX, '').upper())
            if length == 1:
                return predicate_str
            else:
                direction_str = 'I_TURN_{}'.format(tokens[0][1].replace(PREFIX, '').replace(SUFFIX, '').upper())
                if length == 2:
                    repeat = 1
                    after_each = False
                else:
                    manner = tokens[0][2]
                    assert manner in [PREFIX+'opposite'+SUFFIX, PREFIX+'around'+SUFFIX]
                    if manner == PREFIX+'opposite'+SUFFIX:
                        repeat = 2
                        after_each = False
                    else:
                        repeat = 4
                        after_each = True
                if after_each:
                    if predicate != PREFIX+'turn'+SUFFIX:
                        base = '{} {}'.format(direction_str, predicate_str)
                    else:
                        base = direction_str
                    return ' '.join([base]*repeat)
                else:
                    base = ' '.join([direction_str]*repeat)
                    if predicate != PREFIX+'turn'+SUFFIX:
                        return '{} {}'.format(base, predicate_str)
                    else:
                        return base
        elif predicate == PREFIX+'twice'+SUFFIX:
            return ' '.join([tokens[0][1]]*2)
        elif predicate == PREFIX+'thrice'+SUFFIX:
            return ' '.join([tokens[0][1]]*3)
        elif predicate == PREFIX+'and'+SUFFIX:
            return ' '.join(tokens[0][1:])
        elif predicate == PREFIX+'after'+SUFFIX:
            return ' '.join([tokens[0][2], tokens[0][1]])
        else:
            raise Exception('unknown_predicate={}'.format(predicate))


if __name__ == "__main__":

    executor = ProgramExecutorScan()

    file_to_parse = 'data/scan/dsl/simple_split/train.json'
    # file_to_parse = 'data/scan/dsl/simple_split/test.json'
    start_time = time.time()

    with open(file_to_parse, 'r') as data_file:
        for i, line in enumerate(data_file):
            if i % 10000 == 0:
                print('example {}'.format(i))
            print(i)
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question, program, answer = line["question"], line["program"], line["answer"]
            denotation = executor.execute(program)

            if answer != denotation:
                print('example {}'.format(i))
                print(question)
                print(program)
                print(answer)
                print('answer={} denotation={}'.format(answer, denotation))
            # print()

    print("execution took {0} seconds".format(time.time() - start_time))
