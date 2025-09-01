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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory
    from .nopper import Nopper


class Component(ABC):
    """
    Abstract class to contain the common methods and attributes of Blocks and
    Instructions.
    """

    @property
    @abstractmethod
    def duration(self) -> int:
        """Returns the real time duration for the execution of the component in nanoseconds.

        Raises:
            NotImplementedError: Abstract Method.

        Returns:
            int: Real time duration of the component in nanoseconds.
        """
        raise NotImplementedError

    @abstractmethod
    def allocate_registers(self, memory: Memory):
        """Allocates the registers used in this component."""

    @abstractmethod
    def check_nops(self, nopper: Nopper, depth: int):
        """Searches where Nop instructions are needed and saves those positions."""
