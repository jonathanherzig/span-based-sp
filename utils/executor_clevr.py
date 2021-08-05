"""An executor for CLEVR DSL programs."""

import json
from overrides import overrides
from typing import List
import time
from utils import scene_utils
import re

import pyparsing

from utils.executor import Executor


class ClevrEnitity(object):

    def __init__(self, shape: str, color: str, material: str, size: str, left_rank: int, front_rank: int, index: int):
        self.attributes = {
            'shape': shape,
            'color': color,
            'material': material,
            'size': size,
            'left': left_rank,
            'front': front_rank
        }
        self.index = index

    def __eq__(self, other):
        if not isinstance(other, ClevrEnitity):
            # don't compare against something which is not an entity
            return NotImplemented
        return self.index == other.index

    def __hash__(self):
        return int(self.index)


class Scene(object):
    def __init__(self, entities: List[ClevrEnitity]):
        self.entities = entities

    def __init__(self, scene_str: str):
        shapes, materials, sizes, colors, lefts, fronts = scene_utils.parse_scene(scene_str)
        entities = []
        for i, (shape, material, size, color, left, front) in enumerate(zip(shapes, materials, sizes, colors, lefts, fronts)):
            entities.append(ClevrEnitity(shape=shape, color=color, material=material, size=size,
                                         left_rank=left, front_rank=front, index=i))
        self.entities = entities


@Executor.register("clevr_executor")
class ProgramExecutorClevr(Executor):

    def __init__(self):
        self.parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self.parser.setParseAction(self._parse_action)

    @overrides
    def execute(self, program: str, kb_str: str) -> str:

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
        """Executes the program bottom up"""
        predicate = tokens[0][0]
        if predicate == 'scene':
            return Scene(self._scene_str)
        elif predicate == 'filter':
            args = tokens[0][1: -1]
            world = tokens[0][-1]
            return self._filter(args, world)
        elif predicate == 'relate_attribute_equal':
            attribute = tokens[0][1]
            entity_to_relate = tokens[0][2].entities[0]
            return self._relate_attribut_equal(attribute, entity_to_relate)
        elif predicate == 'exist':
            world = tokens[0][1]
            return 'yes' if world.entities else 'no'
        elif predicate == 'query':
            attribute = tokens[0][1]
            entity_to_relate = tokens[0][2].entities[0]
            return entity_to_relate.attributes[attribute]
        elif predicate == 'relate':
            reverse = False
            relation = tokens[0][1]
            if relation=='behind':
                reverse = True
                relation = 'front'
            elif relation == 'right':
                reverse = True
                relation = 'left'
            entity_to_relate = tokens[0][2].entities[0]
            return self._relate(relation, entity_to_relate, reverse)
        elif predicate == 'count':
            world = tokens[0][1]
            return str(len(world.entities))
        elif predicate == 'query_attribute_equal':
            attribute = tokens[0][1]
            entity_1 = tokens[0][2].entities[0]
            entity_2 = tokens[0][3].entities[0]
            return self._query_attribute_equal(attribute,entity_1, entity_2)
        elif predicate == 'union':
            world_1 = tokens[0][1]
            world_2 = tokens[0][2]
            world_1.entities = list(set(world_1.entities + world_2.entities))
            return world_1
        elif predicate == 'intersect':
            world_1 = tokens[0][1]
            world_2 = tokens[0][2]
            world_1.entities = [entity for entity in world_1.entities if entity in world_2.entities]
            return world_1
        elif predicate == 'count_greater':
            world_1 = tokens[0][1]
            world_2 = tokens[0][2]
            match = len(world_1.entities) > len(world_2.entities)
            return self._make_yes_no(match)
        elif predicate == 'count_less':
            world_1 = tokens[0][1]
            world_2 = tokens[0][2]
            match = len(world_1.entities) < len(world_2.entities)
            return self._make_yes_no(match)
        elif predicate == 'count_equal':
            world_1 = tokens[0][1]
            world_2 = tokens[0][2]
            match = len(world_1.entities) == len(world_2.entities)
            return self._make_yes_no(match)
        else:
            raise Exception('unknown_predicate={}'.format(predicate))

    def _make_yes_no(self, bool_res: bool):
        return 'yes' if bool_res else 'no'

    def _filter(self, args: List[str], world: Scene):
        for i, entity in reversed(list(enumerate(world.entities))):
            keep = all([arg in entity.attributes.values() for arg in args])
            if not keep:
                del world.entities[i]
        return world

    def _relate_attribut_equal(self, attribute: str, entity_to_relate: ClevrEnitity):
        scene = Scene(self._scene_str)
        for i, entity in reversed(list(enumerate(scene.entities))):
            keep = entity.attributes[attribute] == entity_to_relate.attributes[attribute] and entity != entity_to_relate
            if not keep:
                del scene.entities[i]
        return scene

    def _relate(self, relation: str, entity_to_relate: ClevrEnitity, reverse: bool):
        scene = Scene(self._scene_str)
        relation_rank = entity_to_relate.attributes[relation]
        for i, entity in reversed(list(enumerate(scene.entities))):
            if not reverse:
                keep = entity.attributes[relation] < relation_rank
            else:
                keep = entity.attributes[relation] > relation_rank
            if not keep:
                del scene.entities[i]
        return scene

    def _query_attribute_equal(self, attribute: str, entity_1: ClevrEnitity, entity_2: ClevrEnitity):
        match = entity_1.attributes[attribute] == entity_2.attributes[attribute]
        return self._make_yes_no(match)


if __name__ == "__main__":

    # Sanity check that all programs execute to the gold answer.
    executor = ProgramExecutorClevr()
    file_to_parse = 'datasets/clevr/dsl/dev.json'
    start_time = time.time()

    with open(file_to_parse, 'r') as data_file:
        for i, line in enumerate(data_file):
            if i % 1000 == 0:
                print('example {}'.format(i))
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question, program, answer, scene = line["question"], line["program"], line["answer"], line["scene"]
            denotation = executor.execute(program, scene)

            if answer != denotation:
                print('example {}'.format(i))
                print(question)
                print(program)
                print(scene)
                print(answer)
                print('answer={} denotation={}'.format(answer, denotation))

    print("execution took {0} seconds".format(time.time() - start_time))
