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

"""Module containing the python representation of all the instructions supported by the Qblox sequencers, as well as
methods to parse instruction strings and convert them into Instruction instances.
"""

from __future__ import annotations

import re
from itertools import compress
from typing import TYPE_CHECKING

from qpysequence.constants import IMMD_MAX_VALUE, IMMD_MIN_VALUE, INST_MIN_WAIT
from qpysequence.enums import InstructionArgument
from qpysequence.program.component import Component
from qpysequence.program.register import Register

if TYPE_CHECKING:
    from qpysequence.program.memory import Memory
    from qpysequence.program.nopper import Nopper


class Instruction(Component):
    """
    Abstract Base Class for all the Instructions.
    """

    def __init__(
        self,
        args: list[int | str | Register] | None = None,
        types: list[list[InstructionArgument]] | None = None,
        bounds: list[tuple[int, int] | None] | None = None,
        duration: int | None = None,
    ):
        super().__init__()
        # PascalCase to snake_case
        self.name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
        self.args: list[int | str | Register] = [] if args is None else args
        self.types: list[list[InstructionArgument]] = [] if types is None else types
        self.bounds: list[tuple[int, int] | None] = [] if bounds is None else bounds
        self.using_registers: bool = False
        self.rtd: int | None = duration
        self.read_registers: set[Register] = set()
        self.write_registers: set[Register] = set()
        self.label: str | None = None
        self.comment: str | None = None
        # Check that all arguments are valid
        self._verify_arguments_type()
        self._verify_optional_registers_consistency()
        # Resolve types list to unique types
        self._determine_arguments_type()
        # Check bounds
        self._check_bounds()
        # Check rtd is multiple of 4
        self._check_rtd()

    @property
    def duration(self) -> int:
        """Returns the duration of the Instruction.

        Returns:
            int: duration of the Instruction.
        """
        return self.rtd or 0

    def _verify_arguments_type(self):
        """Checks that arguments are immediates or a valid registers. Raises an exception otherwise.

        Raises:
            ValueError: Register not valid.
            TypeError: Argument type not valid.
        """
        for arg, arg_type in zip(self.args, self.types):
            if isinstance(arg, int):
                integer_arg: int = arg
                if InstructionArgument.IMMEDIATE not in arg_type or not is_immediate_valid(integer_arg):
                    raise TypeError(f"Argument <{arg}> is not supposed to be an immediate or is not a valid one.")
            elif isinstance(arg, str):
                string_arg: str = arg
                if InstructionArgument.LABEL not in arg_type or not is_label_valid(string_arg):
                    raise TypeError(f"Argument <{arg}> is not supposed to be a label or is not a valid one.")
            elif isinstance(arg, Register):
                if InstructionArgument.REGISTER not in arg_type:
                    raise TypeError(f"Argument <{arg}> is not supposed to be a register.")
            else:
                raise TypeError("Only integer (immediates), string (labels) or Register arguments are supported.")

    def _verify_optional_registers_consistency(self):
        """Checks that all arguments that can be either registers or immediates are of the same type.

        Raises:
            TypeError: All arguments that can be either registers or immediates must be of the same type.
        """
        type_is_ir = [value == "IR" for value in self.types]
        type_is_ir = [
            (InstructionArgument.IMMEDIATE in value and InstructionArgument.REGISTER in value) for value in self.types
        ]
        args_ir = list(compress(self.args, type_is_ir))
        arg_is_immediate = [isinstance(arg, int) for arg in args_ir]
        if any(x != arg_is_immediate[0] for x in arg_is_immediate):
            raise TypeError("All arguments that can be either registers or immediates must be of the same type.")

    def _determine_arguments_type(self):
        """Determines the type of all the arguments."""
        for i, arg in enumerate(self.args):
            if isinstance(arg, int):
                self.types[i] = InstructionArgument.IMMEDIATE
            elif isinstance(arg, str) and is_label_valid(arg):
                self.types[i] = InstructionArgument.LABEL
            elif isinstance(arg, Register):
                self.types[i] = InstructionArgument.REGISTER
            else:
                raise TypeError(f"Argument <{arg}> is not an immediate, label or register.")

    def _check_bounds(self):
        """Checks that all arguments are within bounds if provided.

        Raises:
            ValueError: Argument is out of bounds.
        """
        if len(self.bounds) == 0:
            return
        for i, (arg, arg_type, bounds) in enumerate(zip(self.args, self.types, self.bounds)):
            if arg_type == InstructionArgument.IMMEDIATE and bounds:
                lower, upper = bounds
                if not lower <= arg <= upper:
                    raise ValueError(f"Argument {i} is out of bounds. The valid range is from {lower} to {upper}.")

    def _check_rtd(self):
        """Checks that the real-time duration is equal or greater than 4.

        Raises:
            ValueError: Raised if real-time duration is less than 4.
        """
        if self.rtd and self.rtd < INST_MIN_WAIT:
            raise ValueError(f"Duration of real-time operations have a minimum value of {INST_MIN_WAIT}.")

    def add_read_registers(self, registers: set[Register | int | str]):
        """Checks if the given set of registers to read are actually registers
            and if so adds them to the reading registers set
        Args:
            registers (set[Register  |  int]): set of registers to read
        """
        for reg in registers:
            if isinstance(reg, Register):
                self.read_registers.add(reg)

    def add_write_registers(self, registers: set[Register | int | str]):
        """Checks if the given set of registers to write are actually registers
            and if so adds them to the writing registers set
        Args:
            registers (set[Register  |  int]): set of registers to write
        """
        for reg in registers:
            if isinstance(reg, Register):
                self.write_registers.add(reg)

    def replace_register(self, old: str, new: str) -> None:
        """Replaces all ocurrences of the register `old` by the register `new`.

        Args:
            old (str): Register to replace.
            new (str): New register.
        """
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new

    def __repr__(self) -> str:
        """String representation of the Instruction

        Returns:
            str: Representation of the Instruction.
        """
        label_part = f"{self.label}:" if self.label else ""
        args_part = " " + ", ".join(map(str, self.args)) if self.args else ""
        instruction_part = f"{self.name:<16}{args_part}"
        comment_part = f"# {self.comment}" if self.comment else ""

        # Using formatted string to control spaces and padding
        return f"{label_part:<16}{instruction_part:<32}{comment_part}"

    def allocate_registers(self, memory: Memory):
        """Allocates the registers used by the Instruction.

        Args:
            memory (Memory): Memory object responsible of the allocation.
        """
        for arg in self.args:
            if isinstance(arg, Register):
                memory.allocate_register_and_mark_in_use(arg)

    def check_nops(self, nopper: Nopper, depth: int):
        """Checks if a Nop instruction is required and updates last instruction registers set."""
        nopper.check_intersection(self.read_registers)
        nopper.set_write_registers(self.write_registers)

    def with_label(self, label: str) -> "Instruction":
        """Adds a label to the Instruction.

        Args:
            label (str): The label of the Instruction to add.

        Returns:
            Instruction: The Instruction with the added label.
        """
        self.label = label
        return self

    def with_comment(self, comment: str) -> "Instruction":
        """Adds a comment to the Instruction.

        Args:
            comment (str): The comment of the Instruction to add.

        Returns:
            Instruction: The Instruction with the added comment.
        """
        self.comment = comment
        return self


def is_immediate_valid(immediate: int) -> bool:
    """Checks if immediate is valid.

    Args:
        immediate (int): Immediate value to check

    Returns:
        bool: True if immediate is valid, False otherwise.
    """
    return isinstance(immediate, int) and (IMMD_MIN_VALUE <= immediate <= IMMD_MAX_VALUE)


def is_label_valid(label: str) -> bool:
    """Checks if label is valid.

    Args:
        label (str): Label value to check.

    Returns:
        bool: True if label is valid, False otherwise.
    """
    return isinstance(label, str) and label[0] == "@"
