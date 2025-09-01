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

"""Loop Instruction"""

from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class Loop(Instruction):
    """
    Subtract `count` by one and jump to the instruction indicated by `instr` until `count` reaches zero.

    Args:
        count (Register): Register.
        instr (int | str | Register): Number of the line, or register holding such number, or label to jump to.
    """

    def __init__(self, count: Register, instr: int | str | Register):
        args: list[int | str | Register] = [count, instr]
        types = [
            [InstructionArgument.REGISTER],
            [InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER, InstructionArgument.LABEL],
        ]
        super().__init__(args, types)
        # Add registers to read/write registers sets
        self.add_read_registers({instr, count})
        self.add_write_registers({count})
