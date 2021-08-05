"""A typed grammar for SCAN-SP."""

from overrides import overrides

from treelib import Tree, Node

from span_based.utils.translate_span_trees_to_programs import TreeMapper

PREFIX = 'i_'
SUFFIX = ''


class Entities(object):
    def __init__(self, value):
        self.value = value
        self.return_type = type(self)

    def get_value(self):
        return self.value

    def is_full(self):
        return True  # once created, entities have a valid value


class DirectionEntities(Entities):
    def __init__(self, value):
        super().__init__(value)


class MannerEntities(Entities):
    def __init__(self, value):
        super().__init__(value)


class Predicate(object):
    """Handles an arbitrary predicate which takes up to 2 arguments."""
    def __init__(self, predicate):
        self.predicate = predicate
        self.arg0 = None
        self.arg1 = None

    def get_value(self):
        return self.value

    def _get_full_value(self):
        if self.arg1 is None and self.arg0 is None:
            return '{} ( )'.format(self.predicate)
        elif self.arg1 is None:  # Predicate(Entities)
            return '{} ( {} )'.format(self.predicate, self.arg0.get_value())
        elif self.arg0 is None:  # Predicate(Entities)
            return self.get_value()
        else:  # Predicate(Entities, Entities)
            return '{} ( {} , {} )'.format(self.predicate, self.arg0.get_value(), self.arg1.get_value())

    def apply(self, argument):
        if not issubclass(argument.return_type, self.input_type):
            raise Exception('types do not match {} {}'.format(self.input_type, argument.return_type))
        return self._apply_unary(argument)


class Action(Predicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.return_type = Action
        self.input_type = Entities
        self.value = predicate if '(' in predicate else self._get_full_value()

    def _apply_unary(self, argument):
        assert not self.arg0 or not self.arg1
        if issubclass(type(argument), DirectionEntities):
            assert not self.arg0
            self.arg0 = argument
        else:
            assert(issubclass(type(argument), MannerEntities))
            assert not self.arg1
            self.arg1 = argument
        self.value = self._get_full_value()
        return self

    def is_full(self):
        # does not necessarily takes arguments
        return (self.arg1 is not None and self.arg0 is not None) or self.arg1 is None


class Jump(Action):
    def __init__(self, predicate):
        super().__init__(predicate)


class Run(Action):
    def __init__(self, predicate):
        super().__init__(predicate)


class Look(Action):
    def __init__(self, predicate):
        super().__init__(predicate)


class Turn(Action):
    def __init__(self, predicate):
        super().__init__(predicate)


class Walk(Action):
    def __init__(self, predicate):
        super().__init__(predicate)


class Connective(Predicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.return_type = Entities
        self.input_type = Action
        self.value = predicate

    def _apply_unary(self, argument):
        assert not self.arg0 or not self.arg1
        if not self.arg0:
            self.arg0 = argument
        else:
            assert not self.arg1
            self.arg1 = argument
        if self.is_full():
            return self.return_type(self._get_full_value())
        else:
            return self

    def is_full(self):
        return True if self.arg0 and self.arg1 else False


class And(Connective):
    def __init__(self, predicate):
        super().__init__(predicate)


class After(Connective):
    def __init__(self, predicate):
        super().__init__(predicate)


class UnaryPredicate(Predicate):
    def __init__(self, predicate):
        super().__init__(predicate)
        self.arg1 = '_ignore_'
        self.return_type = Action
        self.input_type = Action
        self.value = predicate

    def _apply_unary(self, argument):
        assert not self.arg0
        self.arg0 = argument
        if self.is_full():
            return self.return_type(self._get_full_value())
        else:
            return self

    def _get_full_value(self):
        assert self.arg0
        return '{} ( {} )'.format(self.predicate, self.arg0.get_value())


    def apply(self, argument):
        # check whether the actual passed type is a subclass of the expected class
        if not issubclass(argument.return_type, self.input_type):
            raise Exception('types do not match {} {}'.format(self.input_type, argument.return_type))
        return self._apply_unary(argument)

    def get_value(self):
        return self.value

    def is_full(self):
        return True if self.arg0 else False


class Twice(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)


class Thrice(UnaryPredicate):
    def __init__(self, predicate):
        super().__init__(predicate)


@TreeMapper.register("scan_tree_mapper")
class TreeMapperScan(TreeMapper):

    def __init__(self):
        self._predicates = {PREFIX+'jump'+SUFFIX: Jump, PREFIX+'run'+SUFFIX: Run, PREFIX+'look'+SUFFIX: Look,
                            PREFIX+'turn'+SUFFIX: Turn,
                            PREFIX+'walk'+SUFFIX: Walk, PREFIX+'and'+SUFFIX: And,
                            PREFIX+'after'+SUFFIX: After, PREFIX+'twice'+SUFFIX: Twice,
                            PREFIX+'thrice'+SUFFIX: Thrice}
        self._entities = {PREFIX+'right'+SUFFIX: DirectionEntities, PREFIX+'left'+SUFFIX: DirectionEntities,
                          PREFIX+'around'+SUFFIX: MannerEntities, PREFIX+'opposite'+SUFFIX: MannerEntities}

        self._node_to_subprog = None

    def _node_to_type(self, node: Node):
        constant = node.data.constant
        if constant in self._predicates:
            return self._predicates[constant](constant)
        elif constant in self._entities:
            return self._entities[constant](constant)
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

        for leaf in tree.leaves():
            span = leaf.data.span
            self._node_to_subprog[span] = self._node_to_type(leaf)
            parent = tree.parent(leaf.identifier)
            if parent and parent not in frontier:
                frontier.append(tree.parent(leaf.identifier))

        while frontier:
            node = frontier.pop()
            children = tree.children(node.identifier)
            assert len(children) == 2
            # check if children were already discovered
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
                    self._node_to_subprog[node.data.span] = self._node_to_type(node)  # ignore children and propagate parent
                else:
                    assert child_2.is_full()  # make sure child_2 value can be formed
                    self._node_to_subprog[node.data.span] = child_1.apply(child_2)
            except Exception as e:
                try:
                    self._node_to_subprog[node.data.span] = child_2.apply(child_1)
                except Exception as e:
                    raise Exception('final apply_exception: {}'.format(e))

            parent = tree.parent(node.identifier)
            if parent and parent not in frontier:
                frontier.insert(0, parent)

        inner_program = self._node_to_subprog[tree.get_node(tree.root).data.span].get_value()  # return the root's value
        return inner_program
