"""
Base class for subcommands under `allennlp.run`.
"""

import argparse
from typing import Callable, Dict, Optional, Type, TypeVar

from radgraph.allennlp.common import Registrable


T = TypeVar("T", bound="Subcommand")


class Subcommand(Registrable):
    """
    An abstract class representing subcommands for allennlp.run.
    If you wanted to (for example) create your own custom `special-evaluate` command to use like

    `allennlp special-evaluate ...`

    you would create a `Subcommand` subclass and then pass it as an override to
    [`main`](#main).
    """

    reverse_registry: Dict[Type, str] = {}

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        raise NotImplementedError

    @classmethod
    def register(
        cls: Type[T], name: str, constructor: Optional[str] = None, exist_ok: bool = False
    ) -> Callable[[Type[T]], Type[T]]:
        super_register_fn = super().register(name, constructor=constructor, exist_ok=exist_ok)

        def add_name_to_reverse_registry(subclass: Type[T]) -> Type[T]:
            subclass = super_register_fn(subclass)
            # Don't need to check `exist_ok`, as it's done by super.
            # Also, don't need to delete previous entries if overridden, they can just stay there.
            cls.reverse_registry[subclass] = name
            return subclass

        return add_name_to_reverse_registry

    @property
    def name(self) -> str:
        return self.reverse_registry[self.__class__]
