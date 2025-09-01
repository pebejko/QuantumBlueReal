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

from .register import Register


class Memory:
    """Memory class to handle and allocate the registers used by a program.

    Args:
        max_registers (int): Maximum number of registers that can be allocated.
    """

    def __init__(self, max_registers: int):
        self.max_registers = max_registers
        self.registry = [False for _ in range(max_registers)]

    def allocate_register_and_mark_in_use(self, register: Register):
        """Allocates the register with the first available number and marks it in use.

        Args:
            register (Register): Register to allocate.

        Raises:
            MemoryError: Reached allocation limit.
        """
        if not register.allocated:
            try:
                index = self.registry.index(False)
            except ValueError as ex:
                raise MemoryError(
                    f"Memory limit exceeded: the maximum number of registers for this memory instance is {self.max_registers}"
                ) from ex

            register.allocate(index)
            self.mark_in_use(register)

    def mark_in_use(self, register: Register):
        """Mark that the register is in use. This will prevent other registers to be allocated with the same number.

        Args:
            register (Register): Register to mark as in-use.
        """
        if register.allocated and 0 <= register.number < self.max_registers:
            self.registry[register.number] = True

    def mark_out_of_use(self, register: Register):
        """Mark that the register is out of use. This will enable other registers to be allocated with the same number.

        Args:
            register (Register): Register to mark as out-of-use.
        """
        if register.allocated and 0 <= register.number < self.max_registers:
            self.registry[register.number] = False
