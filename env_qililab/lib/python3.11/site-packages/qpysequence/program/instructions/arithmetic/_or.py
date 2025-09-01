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

"""Or Instruction"""

from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class Or(Instruction):
    """Bit-wise OR `var` to the value in `origin` and move the result to `destination`.

    Args:
        origin (Register): origin register.
        var (Register | int): value/register to be "ored" to the value in `origin`.
        destination (Register): destination register.
    """

    def __init__(self, origin: Register, var: Register | int, destination: Register):
        args: list[int | str | Register] = [origin, var, destination]
        types = [
            [InstructionArgument.REGISTER],
            [InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER],
            [InstructionArgument.REGISTER],
        ]
        super().__init__(args, types)
        # Add registers to read/write registers sets
        self.add_read_registers({var})
        self.add_write_registers({origin, destination})
