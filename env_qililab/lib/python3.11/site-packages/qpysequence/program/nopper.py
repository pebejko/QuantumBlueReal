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

"""Module for creating Nop instructions"""

from copy import deepcopy

from .register import Register


class Nopper:
    """Class for adding Nop instruction where needed."""

    def __init__(self):
        # Set of the write registers of the last instruction.
        self.write_registers: set[Register] = set()
        # List of indices where a Nop instruction is required.
        self.index_list: list[list[int]] = []
        # Lists to store the current and last instruction indices.
        self.current_block_index: list[int] = [0]
        self.last_block_index: list[int] = [0]
        # Counters to keep track of the numbers of nops in the current block.
        self.current_nop_counter: int = 0
        self.last_nop_counter: int = 0

    def reset(self):
        """Resets all variables."""
        self.write_registers = set()
        self.index_list = []
        self.current_block_index = [0]
        self.last_block_index = [0]
        self.current_nop_counter = 0
        self.last_nop_counter = 0

    def get_list_of_indices(self):
        """Returns a list of indices"""
        return self.index_list

    def set_write_registers(self, registers: set[Register]):
        """Sets as the write_registers atribute the given set of registers.
        Args:
            registers (set[Register]): registers to be interecated with read_registers of the following instruction.
        """
        self.write_registers = registers

    def check_intersection(self, read_registers: set[Register]) -> set[Register]:
        """Check if there's a intersection. If so adds the index to the list and modifies
        both current and last block indices to aviod errors with indices in the same block.

        Args:
            read_registers (set[Register]): set of read registers of the current instruction

        Returns:
            set[Register]: intersection of registers
        """
        if len(read_registers & self.write_registers) > 0:
            self.index_list.append(deepcopy(self.last_block_index))
            self.current_block_index[-1] += 1
            self.last_block_index = deepcopy(self.current_block_index)
            self.current_nop_counter += 1
        return read_registers & self.write_registers

    def update_block_index(self, index: int, depth: int):
        """Updates current and last block index. Depth starts from 1

        Args:
            index (int): index of the current component in the list of components
            depth (int): current depth in nested blocks
        """
        # Updating last attributes with current attributes
        self.last_block_index = deepcopy(self.current_block_index)
        self.last_nop_counter = self.current_nop_counter
        # Entering a new block
        if len(self.current_block_index) < depth:
            # Add one level of depth to the current index
            self.current_block_index.append(index)
            # Reset nop counter
            self.current_nop_counter = 0
        # Moving through the same block
        elif len(self.current_block_index) == depth:
            # Update current block index
            self.current_block_index[depth - 1] = index + self.last_nop_counter
        # Coming off the current block
        else:
            # Delete one or more levels of depth from the current index
            while len(self.current_block_index) > depth:
                self.current_block_index.pop(-1)
            # Update index in current depth
            self.current_block_index[depth - 1] = index
            # Reset nop counter
            self.current_nop_counter = 0
