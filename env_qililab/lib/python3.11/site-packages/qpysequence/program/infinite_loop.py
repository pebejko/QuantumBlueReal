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

from .block import Block
from .instructions import Jmp


class InfiniteLoop(Block):
    """Infinite loop class."""

    def __init__(self, name: str):
        super().__init__(name)
        self._generate_builtin_components()

    def _generate_builtin_components(self):
        """Generates the builtin components of the Loop."""
        self.builtin_components.append(Jmp(f"@{self.name}"))

    @property
    def duration(self) -> int:
        """Duration of all the iterations. Since it is an infinite loop, it returns -1.

        Returns:
            int: Duration in nanoseconds of all the iterations.
        """
        return -1
