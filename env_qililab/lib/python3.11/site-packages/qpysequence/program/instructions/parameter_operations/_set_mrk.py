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

"""SetMrk Instruction"""

from typing import TYPE_CHECKING

from qpysequence.constants import SEQ_N_MARKERS
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction

if TYPE_CHECKING:
    from qpysequence.program.register import Register


class SetMrk(Instruction):
    """Set marker output channels to `marker_outputs` (bits 0-3), where the bit index corresponds to the channel index
    for baseband modules. For QCM-RF module, bit indices 0 & 1 corespond to output enable 1 and 2 respectively; indices
    2 & 3 correspond to marker outputs 1 and 2 respectively. The values are OR'ed by that of other sequencers. The
    parameters are cached and only updated when real_time.io instructions are executed.

    Args:
        marker_outputs (str | int): value/register with a 4-bit integer representing the four marker outputs.
    """

    def __init__(self, marker_outputs: str | int):
        args: list[int | str | Register] = [marker_outputs]
        types = [[InstructionArgument.IMMEDIATE, InstructionArgument.REGISTER]]
        bounds: list[tuple[int, int] | None] = [(0, SEQ_N_MARKERS**2 - 1)]
        super().__init__(args, types, bounds)
