"""An interface for each domain for mapping a span tree to a corresponding program."""

from treelib import Tree

from allennlp.common.registrable import Registrable


class TreeMapper(Registrable):

    def map_tree_to_program(self, tree: Tree) -> str:
        raise NotImplementedError
