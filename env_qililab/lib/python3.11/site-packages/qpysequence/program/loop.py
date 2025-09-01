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
from .component import Component
from .instructions import Add, Jge, Jlt, Move, Nop, Sub
from .instructions import Loop as LoopInstr
from .register import Register


class Loop(Block):
    """Loop class, can hold instructions and other blocks and repeat them for a determinate amount of iterations."""

    def __init__(self, name: str, begin: int, end: int = 0, step: int = -1):
        super().__init__(name)
        self._begin = begin
        self._end = end
        self._step = step
        self._iterations = (end - begin) // step
        self._counter_register = Register()
        self.init_counter_instr = self._init_counter_instruction()
        self.loop_instr = self._generate_loop_instruction()
        self._check_iteration_domain()
        self._generate_builtin_components()

    def _generate_builtin_components(self):
        """Generates the builtin components of the Loop."""
        if incr_instr := self._generate_increment_instruction():
            self.builtin_components.append(incr_instr)
        if not isinstance(self.loop_instr, LoopInstr):
            self.builtin_components.append(Nop())
        self.builtin_components.append(self.loop_instr)

    def _generate_loop_instruction(self) -> LoopInstr | Jge | Jlt:
        """Generates the loop instruction associated with the loop.

        Returns:
            LoopInstr | Jge | Jlt: instruction responsible of controlling the looping behavior.
        """
        if self._end == 0 and self._step == -1:
            return self._generate_std_loop()
        if self._begin > self._end:
            return self._generate_jge_loop()
        return self._generate_jlt_loop()

    def _check_iteration_domain(self):
        """Checks that the values of begin, end and step are compatible.

        Raises:
            ValueError: Combination of begin, end and step are not compatible."""
        if self._step == 0:
            raise ValueError("step = 0 would produce an infinite loop.")
        if self._begin == self._end or ((self._begin < self._end) != (self._step > 0)):
            raise ValueError("Combination of begin, end and step arguments wouldn't produce any iteration.")

    def _generate_std_loop(self) -> LoopInstr:
        """Generates a Loop instruction

        Returns:
            LoopInstr: loop instruction.
        """
        return LoopInstr(self._counter_register, f"@{self.name}")

    def _generate_jge_loop(self) -> Jge:
        """Generates a Jge instruction

        Returns:
            Jge: jge instruction.
        """
        return Jge(self._counter_register, self._end + 1, f"@{self.name}")

    def _generate_jlt_loop(self) -> Jlt:
        """Generates a Jlt instruction

        Returns:
            Jlt: jlt instruction.
        """
        return Jlt(self._counter_register, self._end, f"@{self.name}")

    def _generate_increment_instruction(self) -> Add | Sub | None:
        """Generates the instruction responsible of increasing or decreasing the loop counter if required.

        Returns:
            Add | Sub | None: instruction responsible of increasing or decreasing the counter.
        """
        if isinstance(self.loop_instr, LoopInstr):
            return None
        if isinstance(self.loop_instr, Jlt):
            return Add(self._counter_register, self._step, self._counter_register)
        if isinstance(self.loop_instr, Jge):
            return Sub(self._counter_register, -self._step, self._counter_register)
        raise NotImplementedError("Loop instruction type not supported.")

    @property
    def iterations(self) -> int:
        """Iterations of the loop.

        Returns:
            int: Number of iterations.
        """
        return self._iterations

    @property
    def duration_iter(self) -> int:
        """Duration of a single iteration.

        Returns:
            int: Duration in nanoseconds of a single iteration.
        """
        return super().duration

    @property
    def duration(self) -> int:
        """Duration of all the iterations

        Returns:
            int: Duration in nanoseconds of all the iterations.
        """
        return self.duration_iter * self.iterations

    @property
    def counter_register(self) -> Register:
        """Register used as counter by the loop.

        Returns:
            Register: Register used as counter by the loop.
        """
        return self._counter_register

    def init_counter_register(self):
        """Sets the counter register variable to the value used by the loop instruction."""
        self._counter_register = str(self.loop_instr.args[0])

    def be_appended(self, destination: Block, bot_position: int = 0):
        """Gets appended to the destination block.

        Args:
            destination (Block): Block to be appended to.
            bot_position (int, optional): Position, counting from the last element, to place the loop.
        """
        insert_position = len(destination.components) - bot_position
        destination.components.insert(insert_position, self.init_counter_instr)
        destination.components.insert(insert_position + 1, self)

    def _init_counter_instruction(self) -> Move:
        """Creates the instruction that will initialize the loop counter register.

        Returns:
            Move: move instruction that sets the loop counter register to the value required.
        """
        return Move(self._begin, self.counter_register)

    def append_component(self, component: Component, bot_position: int = 0):
        """Appends a component to the loop at a certain position.

        Args:
            component (Component): Component to append.
            bot_position (int, optional): Position, counting from the last element, to place the component.
                Defaults to 0 (simple append).
        """
        super().append_component(component, bot_position)

    def append_components(self, components: list[Component], bot_position: int = 0):
        """Appends a list of loops to the Block at a certain position.

        Args:
            components ([Component]): list of components to append.
            bot_position (int, optional): Position, counting from the last element, to place the components.
                Defaults to 0 (simple append).
        """
        for component in components:
            self.append_component(component, bot_position=bot_position)
