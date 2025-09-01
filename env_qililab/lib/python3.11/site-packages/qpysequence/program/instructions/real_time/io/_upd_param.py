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

"""UpdParam Instruction"""

from typing import TYPE_CHECKING

from qpysequence.constants import INST_MAX_WAIT, INST_MIN_WAIT
from qpysequence.enums import InstructionArgument
from qpysequence.program.instructions.instruction import Instruction

if TYPE_CHECKING:
    from qpysequence.program.register import Register


class UpdParam(Instruction):
    """
    Update the marker, phase, phase offset, gain and offset parameters set using their respective instructions and then
    wait for `wait_time` number of nanoseconds.

    Args:
        wait_time (int): time to wait in nanoseconds.
    """

    def __init__(self, wait_time: int):
        args: list[int | str | Register] = [wait_time]
        types = [[InstructionArgument.IMMEDIATE]]
        bounds: list[tuple[int, int] | None] = [(INST_MIN_WAIT, INST_MAX_WAIT)]
        super().__init__(args, types, bounds, wait_time)
