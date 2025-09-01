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

"""Wait Instruction"""

from qpysequence.constants import INST_MAX_WAIT, INST_MIN_WAIT
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class Wait(Instruction):
    """
    Wait for `wait_time` number of nanoseconds.

    Args:
        wait_time (Register | int): value/register with the waiting time in nanoseconds.
    """

    def __init__(self, wait_time: Register | int):
        args: list[int | str | Register] = [wait_time]
        types = [[InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER]]
        bounds: list[tuple[int, int] | None] = [(INST_MIN_WAIT, INST_MAX_WAIT)]
        super().__init__(args, types, bounds, wait_time if isinstance(wait_time, int) else None)
        # Add registers to read/write registers sets
        self.add_read_registers({wait_time})
