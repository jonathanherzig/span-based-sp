"""An interface for executing programs."""

from allennlp.common.registrable import Registrable


class Executor(Registrable):

    def execute(self, program: str, kb_str: str = None) -> str:
        raise NotImplementedError
