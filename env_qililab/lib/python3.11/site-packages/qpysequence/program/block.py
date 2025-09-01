# Copyright 2023 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from .component import Component
from .instructions import Instruction

if TYPE_CHECKING:
    from .memory import Memory
    from .nopper import Nopper


class Block(Component):
    """Block class, can hold instructions and other blocks."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.components: list[Component] = []
        self.builtin_components: list[Component] = []

    @property
    def duration(self):
        """Returns the real time duration for the execution of the block.

        Returns:
            int: Real time duration of the block.
        """
        return sum(component.duration for component in self.components)

    def allocate_registers(self, memory: Memory):
        """Allocates the registers used by the Block.

        Args:
            memory (Memory): Memory object responsible of the allocation.
        """
        for component in self.components:
            component.allocate_registers(memory)

    def check_nops(self, nopper: Nopper, depth: int):
        """Iterates through components checking if a Nop instruction is required.

        Args:
            nopper (Nopper): Nopper instance
            depth (int): current depth in nested blocks.
        """
        for index, component in enumerate(self.components):
            nopper.update_block_index(index, depth)
            component.check_nops(nopper, depth + 1)

    def _append_instruction(self, instruction: Instruction, bot_position: int = 0):
        """Appends an instruction to the Block at a certain position.

        Args:
            instruction (Instruction): Component to append.
            bot_position (int, optional): Position, counting from the last element, to place the instruction.
                Defaults to 0 (simple append).
        """
        insert_position = len(self.components) - bot_position
        self.components.insert(insert_position, instruction)

    def _append_block(self, block: Block, bot_position: int = 0):
        """Appends a block to the Block at a certain position.

        Args:
            block (Block): Component to append.
            bot_position (int, optional): Position, counting from the last element, to place the block.
                Defaults to 0 (simple append).
        """
        block.be_appended(self, bot_position)

    def append_component(self, component: Component, bot_position: int = 0):
        """Appends a component to the Block at a certain position.

        Args:
            component (Component): Component to append.
            bot_position (int, optional): Position, counting from the last element, to place the component.
                Defaults to 0 (simple append).
        """
        if isinstance(component, Instruction):
            self._append_instruction(component, bot_position)
        elif isinstance(component, Block):
            self._append_block(component, bot_position)

    def be_appended(self, destination: Block, bot_position: int = 0):
        """Gets appended to the destination block.

        Args:
            destination (Block): Block to be appended to.
            bot_position (int, optional): Position, counting from the last element, to place the block.
        """
        insert_position = len(destination.components) - bot_position
        destination.components.insert(insert_position, self)

    def append_components(self, components: list[Component], bot_position: int = 0):
        """Appends a list of components to the Block at a certain position.

        Args:
            components ([Component]): list of components to append.
            bot_position (int, optional): Position, counting from the last element, to place the components.
                Defaults to 0 (simple append).
        """
        for component in components:
            self.append_component(component, bot_position)

    def __repr__(self) -> str:
        """Returns the string representation of the block.

        Returns:
            str: String representation of the block.
        """
        components_repr = "\n".join(map(repr, self.components))
        builtin_components_repr = "\n".join(map(repr, self.builtin_components))

        return f"{self.name}:\n{components_repr}\n{builtin_components_repr}"
