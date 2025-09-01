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

"""LatchEn Instruction"""

from qpysequence.constants import INST_MAX_WAIT, INST_MIN_WAIT, MASK_MAX_VALUE
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class SetLatchEn(Instruction):
    """
    Enable/disable all trigger network address counters based on the `enable` argument and then wait for `wait_time`
    number of nanoseconds. Once enabled, the trigger network address counters will count all triggers on the trigger
    network. When disabled, the counters hold their last values.

    Args:
        enable (Register | int): trigger mask bits 0-15.
        wait_time (int): waiting time in nanoseconds.
    """

    def __init__(self, enable: Register | int, wait_time: int):
        args: list[int | str | Register] = [enable, wait_time]
        types = [[InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER], [InstructionArgument.IMMEDIATE]]
        bounds: list[tuple[int, int] | None] = [(0, MASK_MAX_VALUE), (INST_MIN_WAIT, INST_MAX_WAIT)]
        super().__init__(args, types, bounds, wait_time)
        # Add registers to read/write registers sets
        self.add_read_registers({enable})
