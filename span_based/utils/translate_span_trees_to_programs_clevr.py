"""A typed grammar for CLEVR."""

from functools import singledispatch
from overrides import overrides

from span_based.utils.translate_span_trees_to_programs import TreeMapper

from treelib import Tree, Node


class Attribute(object):
    def __init__(self, attribute):
        self.value = attribute

    def is_full(self):
        return False


class Outermost(object):
    def __init__(self, attribute):
        self.value = attribute

    def is_full(self):
        return True


class Entities(object):
    def __init__(self, attribute_value, is_accum=False):
        self.value = attribute_value
        self.is_accum = is_accum  # Whether accumulating attribute values ('red', 'small' etc.)

    def apply(self, entities):
        if not isinstance(entities, Entities):
            raise Exception('types do not match {} {}'.format(type(self), type(entities)))
        assert self.is_accum or entities.is_accum
        if self.is_accum and entities.is_accum:
            self.value = '{} , {}'.format(self.value, entities.value)
        else:
            if self.is_accum and not entities.is_accum:
                self.value = 'filter ( {} , {} )'.format(self.value, entities.value)
            else:
                self.value = 'filter ( {} , {} )'.format(entities.value, self.value)
            self.is_accum = False
        return self

    def close_accum(self):
        self.value = 'filter ( {} , scene ( ) )'.format(self.value)
        self.is_accum = False

    def is_full(self):
        return not self.is_accum


class Relation(object):
    def __init__(self, relation):
        self.value = relation

    def apply(self, entities: Entities):
        if entities.is_accum:
            entities.close_accum()
        return Entities('relate ( {} , {} )'.format(self.value, entities.value))


class AttributeEntitiesPredicate(object):
    """Handles an arbitrary predicate which takes up to 3 arguments."""
    def __init__(self, predicate):
        self.value = predicate
        self.apply = singledispatch(self.apply)
        self.apply.register(Attribute, self._apply_attribute)
        self.apply.register(Entities, self._apply_entities)
        self.arg0 = None
        self.arg1 = None
        self.arg2 = None
        self.return_type = None  # The type the specific instance returns

    def _get_full_value(self):
        if self.arg0 == '_ignore_' and self.arg2 == '_ignore_':  # Predicate(Entities)
            return '{} ( {} )'.format(self.value, self.arg1.value)
        elif self.arg0 == '_ignore_':  # Predicate(Entities, Entities)
            return '{} ( {} , {} )'.format(self.value, self.arg1.value, self.arg2.value)
        elif self.arg2 == '_ignore_':  # Predicate(X, Entities) where X in {Attribute, Relation}
            return '{} ( {} , {} )'.format(self.value, self.arg0.value, self.arg1.value)
        else:  # Predicate(X, Entities, Entities) where X in {Attribute, Relation}
            return '{} ( {} , {} , {} )'.format(self.value, self.arg0.value, self.arg1.value, self.arg2.value)

    def apply(self, s):
        raise TypeError("This type isn't supported: {}".format(type(s)))

    def _apply_attribute(self, attribute: Attribute):
        assert not self.arg0
        self.arg0 = attribute
        # self.is_full = True if self.arg0 and self.arg1 and self.arg2 else False
        if self.is_full():
            return self.return_type(self._get_full_value())
        else:
            return self

    def _apply_entities(self, entities: Entities):
        assert not self.arg1 or not self.arg2
        if entities.is_accum:
            entities.close_accum()
        if not self.arg1:
            self.arg1 = entities
        else:
            self.arg2 = entities
        # self.is_full = True if self.arg0 and self.arg1 and self.arg2 else False
        if self.is_full():
            return self.return_type(self._get_full_value())
        else:
            return self

    def is_full(self):
        return True if self.arg0 and self.arg1 and self.arg2 else False


class Exist(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.arg2 = '_ignore_'
        self.return_type = Outermost


class Count(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.arg2 = '_ignore_'
        self.return_type = Outermost


class RelateAttributeEqual(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg2 = '_ignore_'
        self.return_type = Entities


class Query(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg2 = '_ignore_'
        self.return_type = Entities


class Union(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.return_type = Entities


class Intersect(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.return_type = Entities


class QueryAttributeEqual(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.return_type = Outermost


class CountGreater(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.return_type = Entities


class CountLess(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.return_type = Entities


class CountEqual(AttributeEntitiesPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg0 = '_ignore_'
        self.return_type = Entities


@TreeMapper.register("clevr_tree_mapper")
class TreeMapperClevr(TreeMapper):

    def __init__(self):
        self._predicates = {'exist': Exist, 'relate_attribute_equal': RelateAttributeEqual, 'query': Query,
                            'query_attribute_equal': QueryAttributeEqual, 'count': Count, 'count_greater': CountGreater,
                            'count_less': CountLess, 'count_equal': CountEqual,
                            'union': Union, 'intersect': Intersect}
        self._attributes = {'shape', 'color', 'size', 'material'}
        self._attribute_values = {'large', 'small',
                                  "cube", "sphere", "cylinder",
                                  "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
                                  "rubber", "metal"}
        self._relations = {"left", "right", "behind", "front"}
        self._node_to_subprog = None

    def _node_to_type(self, node: Node):
        constant = node.data.constant
        if constant in self._predicates:
            return self._predicates[constant](constant)
        elif constant in self._attributes:
            return Attribute(constant)
        elif constant in self._attribute_values:
            return Entities(constant, is_accum=True)
        elif constant in self._relations:
            return Relation(constant)
        elif constant == 'NO-LABEL':
            return None
        elif constant == 'span':
            return None
        else:
            raise Exception('unknown_constant={}'.format(constant))

    @overrides
    def map_tree_to_program(self, tree: Tree) -> str:

        self._node_to_subprog = {}
        frontier = []  # Tree nodes that are left to be explored

        # Builds the program for each node by traversing bottom-up starting from the leaves, going
        # next to their parents etc..
        for leaf in tree.leaves():
            span = leaf.data.span
            self._node_to_subprog[span] = self._node_to_type(leaf)
            parent = tree.parent(leaf.identifier)
            if parent not in frontier:
                frontier.append(tree.parent(leaf.identifier))

        while frontier:
            node = frontier.pop()
            children = tree.children(node.identifier)
            assert len(children) == 2
            if not all([child.data.span in self._node_to_subprog for child in children]):
                frontier.insert(0, node)
                continue

            child_1 = self._node_to_subprog[children[0].data.span]
            child_2 = self._node_to_subprog[children[1].data.span]
            try:
                if child_1 and not child_2:  # child_2=='NO_LABEL'
                    self._node_to_subprog[node.data.span] = child_1
                elif not child_1 and child_2:  # child_1=='NO_LABEL'
                    self._node_to_subprog[node.data.span] = child_2
                elif not child_1 and not child_2:  # Both children are assigned with 'NO_LABEL'
                    self._node_to_subprog[node.data.span] = self._node_to_type(
                        node)  # ignore children and propagate parent
                else:
                    self._node_to_subprog[node.data.span] = child_1.apply(child_2)
            except Exception as e:
                try:
                    self._node_to_subprog[node.data.span] = child_2.apply(child_1)
                except Exception as e:
                    raise Exception('final apply_exception: {}'.format(e))

            parent = tree.parent(node.identifier)
            if parent and parent not in frontier:
                frontier.insert(0, parent)

        root_prog = self._node_to_subprog[tree.get_node(tree.root).data.span]
        if not root_prog.is_full():
            raise Exception('top value is not full')
        return root_prog.value  # return the root's value
