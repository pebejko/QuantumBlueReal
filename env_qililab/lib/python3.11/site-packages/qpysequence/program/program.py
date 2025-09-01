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

"""Program module that holds the python representation of all the components in a Q1ASM program.

Raises:
    MemoryError: Reached limit of registers usage.
    KeyError: Attempted to add a block to the program named as another block already present.
    NotFoundErr: Block not found.
"""

from typing import List
from xml.dom import NotFoundErr

from qpysequence.constants import PROG_MAX_REGISTERS

from .block import Block
from .instructions import Nop, WaitSync
from .loop import Loop
from .memory import Memory
from .nopper import Nopper


class Program:
    """Program class that holds the python representation of all the components in a Q1ASM program.

    Args:
        wait_sync (bool): Add (True) or not (False) a wait_sync instruction to the setup block.
    """

    def __init__(self, wait_sync: bool = True):
        self._wait_sync = wait_sync
        self.blocks: List[Block] = []
        self._init_setup(wait_sync)
        self.repr = None
        self._memory = Memory(PROG_MAX_REGISTERS)
        self._nopper: Nopper = Nopper()
        self._compiled = False

    def _init_setup(self, wait_sync: bool) -> None:
        """Initializes the setup block."""
        setup = Block("setup")
        if wait_sync:
            setup.append_component(WaitSync(4))
        self.append_block(setup)

    @property
    def duration(self) -> int:
        """Returns the real time duration for the whole program execution.

        Returns:
            int: Duration of the program in nanoseconds.
        """
        return sum(b.duration for b in self.blocks)

    def append_block(self, block: Block) -> None:
        """Appends `block` to the program.

        Args:
            block (Block): Block to be appended.
        """
        self._check_block_name(block)
        if isinstance(block, Loop):
            self._append_loop(block)
        else:
            self.blocks.append(block)

    def _append_loop(self, loop: Loop) -> None:
        """Appends `loop` to the program.

        Args:
            loop (Loop): Loop to be appended.
        """
        setup_block = self.get_block("setup")
        setup_block.append_component(loop.init_counter_instr, bot_position=1 if self._wait_sync else 0)
        self.blocks.append(loop)

    def _check_block_name(self, block: Block) -> None:
        """Checks if the `block` name is already present in the program and raises a KeyError if so.

        Args:
            block (Block): block to check the name.

        Raises:
            KeyError: A block with the name `block.name` already exists in the program.
        """
        for existing_block in self.blocks:
            if block.name == existing_block.name:
                raise KeyError(f"A Block with the name {block.name} already exists in this program.")

    def get_block(self, name: str) -> Block:
        """Returns a block named `name` if it exists in the program.

        Args:
            name (str): Name of the block

        Raises:
            NotFoundErr: Block with name `name` does not exist.

        Returns:
            Block: Block named `name`.
        """
        for block in self.blocks:
            if block.name == name:
                return block
        raise NotFoundErr(f"Block with name {name} does not exist.")

    def allocate_registers(self):
        """Allocates all registers used by the program."""
        for block in self.blocks:
            block.allocate_registers(self._memory)

    def check_nops(self):
        """Searches where Nop instructions are needed"""
        for index, block in enumerate(self.blocks):
            self._nopper.update_block_index(index=index, depth=1)
            block.check_nops(self._nopper, depth=2)

    def place_nops(self):
        """Places Nop instrctions where needed."""
        self._nopper.reset()
        self.check_nops()
        block = Block("noname")
        # For each index in nopper's indices list
        for index in self._nopper.get_list_of_indices():
            # Counter to control how we are moving through nested blocks
            cont = 1
            # Get the block of the current index
            block = self.blocks[index[cont - 1]]
            while cont < len(index) - 1:
                block = block.components[index[cont]]
                cont += 1
            # Append a nop instruction minding the position starts from the end
            block.append_component(Nop(), len(block.components) - index[cont] - 1)

    def compile(self):
        """Compiles the program."""
        self.allocate_registers()
        self.place_nops()
        self._compiled = True

    def __repr__(self) -> str:
        """Returns the string Q1ASM representation of the program.

        Returns:
            str: Q1ASM representation of the program.
        """
        if not self._compiled:
            self.compile()
        return "".join(repr(block) + "\n" for block in self.blocks)
