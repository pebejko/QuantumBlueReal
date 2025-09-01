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

"""Play Instruction"""

from qpysequence.constants import INST_MAX_WAIT, INST_MIN_WAIT
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class Play(Instruction):
    """
    Update the marker, phase, phase offset, gain and offset parameters set using their respective instructions, start
    playing AWG waveforms stored at indexes `waveform_0` on path 0 and `waveform_1` on path 1 and finallywait for
    `wait_time` (default is 4) number of nanoseconds. The arguments are either all set through immediates or registers.

    Args:
        waveform_0 (Register | int): value/register with the index of the waveform for path 0.
        waveform_0 (Register | int): value/register with the index of the waveform for path 1.
        wait_time (int, optional): time to wait in nanoseconds. Defaults to 4.
    """

    def __init__(self, waveform_0: Register | int, waveform_1: Register | int, wait_time: int = INST_MIN_WAIT):
        args: list[int | str | Register] = [waveform_0, waveform_1, wait_time]
        types = [
            [InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER],
            [InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER],
            [InstructionArgument.IMMEDIATE],
        ]
        bounds: list[tuple[int, int] | None] = [None, None, (INST_MIN_WAIT, INST_MAX_WAIT)]
        super().__init__(args, types, bounds, wait_time)
        # Add registers to read/write registers sets
        self.add_read_registers({waveform_0, waveform_1})
