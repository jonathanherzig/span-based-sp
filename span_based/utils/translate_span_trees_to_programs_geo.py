"""A typed grammar for GeoQuery."""

from overrides import overrides

from treelib import Tree, Node

from span_based.utils.translate_span_trees_to_programs import TreeMapper


class Entities(object):
    def __init__(self, value, category=None):
        self.value = value
        self.category = category
        self.return_type = type(self)

    def get_value(self):
        if self.category:
            return '{} ( {} )'.format(self.category, self.value)
        else:
            return self.value

    def is_full(self):
        return True  # once created, entities have a valid value


class NamedEntities(Entities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class StateCityEntities(NamedEntities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class CityEntities(StateCityEntities):
    def __init__(self, value, category=None):
        super().__init__(value, category)

    def get_value(self):
        if self.category:
            return '{} ( {}, _ )'.format(self.category, self.value)
        else:
            return self.value


class StateEntities(StateCityEntities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class PlaceEntities(NamedEntities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class RiverEntities(NamedEntities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class NumericalEntities(Entities):
    def __init__(self, value, category=None):
        super().__init__(value, category)


class Predicate(object):
    """Handles an arbitrary predicate which takes up to 2 arguments."""
    def __init__(self, predicate):
        self.value = predicate
        self.arg0 = None
        self.arg1 = None
        self.return_type = None  # The type the specific instance returns

    def get_value(self):
        return self.value

    def _get_full_value(self):
        if self.arg1 == '_ignore_':  # Predicate(Entities)
            return '{} ( {} )'.format(self.get_value(), self.arg0.get_value())
        else:  # Predicate(Entities, Entities)
            return '{} ( {} , {} )'.format(self.get_value(), self.arg0.get_value(), self.arg1.get_value())

    def is_full(self):
        return self.arg0 and self.arg1


class Exclude(Predicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.return_type = None
        self.input_type = NamedEntities

    def _apply_unary(self, argument):
        assert not self.arg0 or not self.arg1
        if not issubclass(type(argument), NamedEntities):  # it is the first argument, e.g. river(all)
            assert not self.arg0
            self.arg0 = argument
            self.return_type = argument.return_type
        else:
            assert not self.arg1
            self.arg1 = argument
        self.is_full = True if self.arg0 and self.arg1 else False
        if self.is_full:
            return self.return_type(self._get_full_value())
        else:
            return self

    def apply(self, argument):
        if not issubclass(argument.return_type, self.input_type):
            raise Exception('types do not match {} {}'.format(self.input_type, argument.return_type))
        return self._apply_unary(argument)


class UnaryPredicate(Predicate):
    def __init__(self, predicate, input_type, return_type):
        super().__init__(predicate)
        self.arg1 = '_ignore_'
        self.return_type = return_type
        self.input_type = input_type

    def _apply_unary(self, argument):
        assert not self.arg0
        self.arg0 = argument
        self.is_full = True if self.arg0 and self.arg1 else False
        if self.is_full:
            return self.return_type(self._get_full_value())
        else:
            return self

    def apply(self, argument):
        # check whether the actual passed type is a subclass of the expected class
        if not issubclass(argument.return_type, self.input_type):
            raise Exception('types do not match {} {}'.format(self.input_type, argument.return_type))
        return self._apply_unary(argument)


class City(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, CityEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class Loc1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, StateEntities)


class HighPoint1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateEntities, PlaceEntities)


class State(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, StateEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class River(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, RiverEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class Capital(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, CityEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class Capital1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateEntities, CityEntities)


class Capital2(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, CityEntities, StateEntities)


class Highest(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, PlaceEntities, PlaceEntities)


class Place(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, PlaceEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class Mountain(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, PlaceEntities)

    def get_value(self):
        if not self.arg0:
            return '{} ( all )'.format(self.value)  # close predicate if no argument exists
        else:
            return self.value

    def is_full(self):
        return True


class Lake(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, PlaceEntities)


class Higher2(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, PlaceEntities, PlaceEntities)


class Longest(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, RiverEntities, RiverEntities)


class Traverse1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, RiverEntities, StateEntities)


class Elevation1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, PlaceEntities, NumericalEntities)


class Len(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, RiverEntities, NumericalEntities)


class Shortest(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, RiverEntities, RiverEntities)


class Lowest(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, PlaceEntities, PlaceEntities)


class Count(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, NumericalEntities)


class Density1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateCityEntities, NumericalEntities)


class NextTo1(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateEntities, StateEntities)


class Sum(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NumericalEntities, NumericalEntities)


class LargestOne(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NumericalEntities, StateCityEntities)


class SmallestOne(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, NumericalEntities, StateCityEntities)


class Fewest(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateEntities, StateEntities)


class Largest(UnaryPredicate):
    """Can operate on states or cities"""

    def __init__(self, predicate):
        super().__init__(predicate, None, None)

    def apply(self, argument):
        if argument.return_type != StateEntities and argument.return_type != CityEntities:
            raise Exception('types do not match {} {}'.format('not StateEntities or CityEntities', argument.return_type))
        self.return_type = argument.return_type  # returns the same type as the argument's type
        return self._apply_unary(argument)


class Traverse2(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateCityEntities, RiverEntities)


class Loc2(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate, StateCityEntities, NamedEntities)


class Smallest(UnaryPredicate):
    """Can operate on states or cities"""

    def __init__(self, predicate):
        super().__init__(predicate, None, None)

    def apply(self, argument):
        if argument.return_type != StateEntities and argument.return_type != CityEntities and argument.return_type != NumericalEntities:
            raise Exception('types do not match {} {}'.format('not StateEntities or CityEntities', argument.return_type))
        self.return_type = argument.return_type  # returns the same type as the argument's type
        return self._apply_unary(argument)


class Size(UnaryPredicate):
    """Can operate on states or cities"""
    def __init__(self, predicate):
        super().__init__(predicate, None, NumericalEntities)

    def apply(self, argument):
        if argument.return_type != StateEntities and argument.return_type != CityEntities:
            raise Exception('types do not match {} {}'.format('not StateEntities or CityEntities', argument.return_type))
        return self._apply_unary(argument)


class NextTo2(UnaryPredicate):
    """Can operate on states or rivers"""
    def __init__(self, predicate):
        super().__init__(predicate, NamedEntities, StateEntities)


class Area1(UnaryPredicate):
    """Can operate on states or cities"""
    def __init__(self, predicate):
        super().__init__(predicate, StateCityEntities, NumericalEntities)


class Population1(UnaryPredicate):
    """Can operate on states or cities"""
    def __init__(self, predicate):
        super().__init__(predicate, StateCityEntities, NumericalEntities)


class Major(UnaryPredicate):
    """Can operate on cities, rivers or lakes"""

    def __init__(self, predicate):
        super().__init__(predicate, None, None)

    def apply(self, argument):
        if argument.return_type != RiverEntities and argument.return_type != CityEntities and argument.return_type != PlaceEntities:
            raise Exception('types do not match {} {}'.format('not StateEntities or CityEntities', argument.return_type))
        self.return_type = argument.return_type  # returns the same type as the argument's type
        return self._apply_unary(argument)


class Most(UnaryPredicate):
    """Can operate on states or rivers"""

    def __init__(self, predicate):
        super().__init__(predicate, None, None)

    def apply(self, argument):
        if argument.return_type != RiverEntities and argument.return_type != StateEntities:
            raise Exception('types do not match {} {}'.format('not StateEntities or RiverEntities',
                                                              argument.return_type))
        self.return_type = argument.return_type  # returns the same type as the argument's type
        return self._apply_unary(argument)


@TreeMapper.register("geo_tree_mapper")
class TreeMapperGeo(TreeMapper):

    def __init__(self):
        self._predicates = {'city': City, 'loc_2': Loc2, 'high_point_1': HighPoint1, 'next_to_2': NextTo2,
                            'state': State, 'river': River, 'capital': Capital, 'highest': Highest, 'place': Place,
                            'lake': Lake, 'largest': Largest, 'longest': Longest, 'traverse_2': Traverse2,
                            'size': Size, 'elevation_1': Elevation1, 'len': Len, 'shortest': Shortest, 'count': Count,
                            'major': Major, 'population_1': Population1, 'smallest': Smallest, 'loc_1': Loc1,
                            'exclude': Exclude, 'area_1': Area1, 'next_to_1': NextTo1, 'traverse_1': Traverse1,
                            'sum': Sum, 'density_1': Density1, 'mountain': Mountain, 'largest_one': LargestOne,
                            'capital_1': Capital1, 'lowest': Lowest, 'smallest_one': SmallestOne, 'capital_2': Capital2,
                            'higher_2': Higher2, 'most': Most, 'fewest': Fewest}
        self._entities = {'stateid': StateEntities, 'countryid': StateEntities, 'cityid': CityEntities,
                          'placeid': PlaceEntities, 'riverid': RiverEntities}

        self._node_to_subprog = None

    def _node_to_type(self, node: Node):
        constant = node.data.constant
        if '#' in constant:  # This indicates an entity:
            constant, value = constant.split('#')
        if constant in self._predicates:
            return self._predicates[constant](constant)
        elif constant in self._entities:
            return self._entities[constant](value, constant)
        elif constant == 'NO-LABEL':
            return None
        elif constant == 'span':
            return None
        else:
            raise Exception('unknown_constant={}'.format(constant))

    def merge_children(self, child_1, child_2, node):
        try:
            if child_1 and not child_2:  # child_2=='NO_LABEL'
                result = child_1
            elif not child_1 and child_2:  # child_1=='NO_LABEL'
                result = child_2
            elif not child_1 and not child_2:  # Both children are assigned with 'NO_LABEL'
                result = self._node_to_type(
                    node)  # ignore children and propagate parent
            else:
                assert child_2.is_full()  # make sure child_2 value can be formed
                result = child_1.apply(child_2)
        except Exception as e:
            try:
                result = child_2.apply(child_1)
            except Exception as e:
                raise Exception('final apply_exception: {}'.format(e))
        return result

    @overrides
    def map_tree_to_program(self, tree: Tree) -> str:

        self._node_to_subprog = {}

        frontier = []  # Tree nodes that are left to be explored

        for leaf in tree.leaves():
            span = leaf.data.span
            self._node_to_subprog[span] = self._node_to_type(leaf)
            parent = tree.parent(leaf.identifier)
            if parent and parent not in frontier:
                frontier.append(tree.parent(leaf.identifier))

        while frontier:
            node = frontier.pop()
            children = tree.children(node.identifier)
            assert len(children) in [2, 3]
            # check if children were already discovered
            if not all([child.data.span in self._node_to_subprog for child in children]):
                frontier.insert(0, node)
                continue

            if len(children) == 2:
                child_1 = self._node_to_subprog[children[0].data.span]
                child_2 = self._node_to_subprog[children[1].data.span]
                self._node_to_subprog[node.data.span] = self.merge_children(child_1, child_2, node)
            else:
                children.sort(key=lambda c: c.data.span[0])
                child_1 = self._node_to_subprog[children[0].data.span]
                child_2 = self._node_to_subprog[children[1].data.span]
                child_3 = self._node_to_subprog[children[2].data.span]
                intermediate = self.merge_children(child_1, child_3, node)
                self._node_to_subprog[node.data.span] = self.merge_children(child_2, intermediate,
                                                                            node)
            parent = tree.parent(node.identifier)
            if parent and parent not in frontier:
                frontier.insert(0, parent)

        inner_program = self._node_to_subprog[tree.get_node(tree.root).data.span].get_value()  # return the root's value
        return 'answer ( {} )'.format(inner_program)
