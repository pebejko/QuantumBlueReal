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

from .arithmetic import Add, And, Asl, Asr, Move, Not, Or, Sub, Xor
from .conditional import SetCond
from .control import Illegal, Nop, Stop
from .instruction import Instruction
from .jumps import Jge, Jlt, Jmp, Loop
from .parameter_operations import ResetPh, SetAwgGain, SetAwgOffs, SetFreq, SetMrk, SetPh, SetPhDelta
from .real_time import (
    Acquire,
    AcquireTtl,
    AcquireWeighed,
    LatchRst,
    Play,
    SetLatchEn,
    UpdParam,
    Wait,
    WaitSync,
    WaitTrigger,
)

__all__ = [
    "Acquire",
    "AcquireTtl",
    "AcquireWeighed",
    "Add",
    "And",
    "Asl",
    "Asr",
    "Illegal",
    "Instruction",
    "Jge",
    "Jlt",
    "Jmp",
    "LatchRst",
    "Loop",
    "Move",
    "Nop",
    "Not",
    "Or",
    "Play",
    "ResetPh",
    "SetAwgGain",
    "SetAwgOffs",
    "SetCond",
    "SetFreq",
    "SetLatchEn",
    "SetMrk",
    "SetPh",
    "SetPhDelta",
    "Stop",
    "Sub",
    "UpdParam",
    "Wait",
    "WaitSync",
    "WaitTrigger",
    "Xor",
]
