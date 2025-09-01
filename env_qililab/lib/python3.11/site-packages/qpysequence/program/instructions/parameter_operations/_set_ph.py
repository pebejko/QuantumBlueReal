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

"""SetPh Instruction"""

from qpysequence.constants import NCO_MAX_INT_PHASE
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction
from qpysequence.program.register import Register


class SetPh(Instruction):
    """
    Set the relative `phase` of the NCO used by the AWG and acquisition. The phase is divided into 1e9 steps between 0º
    and 360º, expressed as an integer between 0 and 1e9 (e.g. 45º=125e6). The phase parameter is cached and only
    updated when the `upd_param`, `play`, `acquire` or `acquire_weighted` instructions are executed.

    Args:
        phase (int | Register): integer between 0 and 1e9 representing the NCO relative phase from 0º to 360º.
    """

    def __init__(self, phase: int | Register):
        args: list[int | str | Register] = [phase]
        types = [[InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER]]
        bounds: list[tuple[int, int] | None] = [(0, NCO_MAX_INT_PHASE)]
        super().__init__(args, types, bounds)
        # Add registers to read/write registers sets
        self.add_read_registers({phase})
