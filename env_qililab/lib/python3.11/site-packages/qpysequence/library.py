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

from math import pi

from qpysequence.constants import (
    AWG_MAX_GAIN,
    INST_MAX_WAIT,
    INST_MIN_WAIT,
    NCO_HZ_TO_INT_MULTIPLIER,
    NCO_MAX_HZ_FREQ,
    NCO_MAX_INT_PHASE,
    NCO_MIN_HZ_FREQ,
    SEQ_WAIT_STEP,
)
from qpysequence.program.block import Block
from qpysequence.program.instructions import Add, And, Asl, Asr, Jge, Jlt, Move, Nop, SetAwgGain, SetFreq, SetPh, Wait
from qpysequence.program.loop import Loop
from qpysequence.program.register import Register


def counted(function):
    """Counting decorator to track how many times a function has been called."""

    def wrapped(*args, **kwargs):
        """Increments the `calls` attribute of the method in one.

        Returns:
            function: input function with `calls` attribute incremented.
        """
        wrapped.calls += 1
        return function(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


@counted
def long_wait(wait_time: int, round_type: str | None = None) -> Block:
    """Returns a block with an arbitrary waiting time.

    Args:
        wait_time (int): time to wait.
        round_type (str): type of rounding. Valid values are "nearest", "up" and "down". Defaults to None.

    Returns:
        long_wait_block (Block): Block containing a waiting loop or single wait instruction if wait_time is small.
    """
    # Find INST_MAX_WAIT wait loop iterations and remaining time
    max_wait_iters = wait_time // INST_MAX_WAIT
    remaining_time = wait_time - max_wait_iters * INST_MAX_WAIT
    # Round remaining time if required
    round_time = SEQ_WAIT_STEP * round(remaining_time / SEQ_WAIT_STEP)
    if round_type == "nearest":
        remaining_time = round_time
    elif round_type == "up":
        remaining_time = round_time if round_time > remaining_time else round_time + SEQ_WAIT_STEP
    elif round_type == "down":
        remaining_time = round_time if round_time < remaining_time else round_time - SEQ_WAIT_STEP
    # Long Wait block
    block_name = f"long_wait_{long_wait.calls}"
    long_wait_block = Block(block_name)
    # Wait loop and single wait instruction
    if max_wait_iters > 0:
        wait_loop = Loop(f"{block_name}_loop", max_wait_iters)
        wait_loop.append_component(Wait(INST_MAX_WAIT))
        long_wait_block.append_component(wait_loop)
    # Only add the single wait instruction if the remaining time is larger than zero
    if remaining_time >= INST_MIN_WAIT:
        single_wait_instr = Wait(remaining_time)
        long_wait_block.append_component(single_wait_instr)
    return long_wait_block


def set_phase_rad(rads: float) -> SetPh:
    """Returns a SetPh instruction with its arguments converted using the `rads` parameter.

    Args:
        rads (float): Phase in radiants.

    Returns:
        SetPh: equivalent SetPh instruction.
    """
    rads %= 2 * pi
    q1asm_phase = int(rads * NCO_MAX_INT_PHASE / (2 * pi))
    return SetPh(q1asm_phase)


def set_freq_hz(freq: float) -> SetFreq:
    """Returns a SetFreq instruction with its arguments converted using the `freq` parameter.

    Args:
        rads (float): Frequency in Hz.

    Returns:
        SetFreq: equivalent SetFreq instruction.
    """
    if not (NCO_MIN_HZ_FREQ <= freq <= NCO_MAX_HZ_FREQ):
        raise ValueError(f"Frequency out of range [{NCO_MIN_HZ_FREQ}, {NCO_MAX_HZ_FREQ}].")

    q1asm_freq = int(freq * NCO_HZ_TO_INT_MULTIPLIER)
    return SetFreq(q1asm_freq)


def set_awg_gain_relative(gain_0: float, gain_1: float) -> SetAwgGain:
    """Returns a SetAwgGain Instruction from a relative gain value.

    Args:
        gain_0 (float): Gain from -1.0 to 1.0 for path 0.
        gain_1 (float): Gain from -1.0 to 1.0 for path 1.

    Returns:
        SetAwgGain: SetAwgGain Instruction.
    """
    if not (-1.0 <= gain_0 <= 1.0) or not (-1.0 <= gain_1 <= 1.0):
        raise ValueError("Gain must be between -1.0 and 1.0")
    steps_0 = int(gain_0 * AWG_MAX_GAIN)
    steps_1 = int(gain_1 * AWG_MAX_GAIN)
    return SetAwgGain(steps_0, steps_1)


def multiply_counter(function):
    """Counting decorator to track how many times a function has been called."""

    def wrapped(*args, **kwargs):
        """Increments the `calls` attribute of the method in one.

        Returns:
            function: input function with `calls` attribute incremented.
        """
        wrapped.calls += 1
        return function(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


@multiply_counter
def multiply(
    a: Register,
    b: Register,
    target: Register,
    reg_aux_1: Register = Register(),
    reg_aux_2: Register = Register(),
    reg_aux_3: Register = Register(),
) -> Block:
    """Creates the Q1ASM program needed to multiply two integers.

    Args:
        a (Register): first integer stored in a register.
        b (Register): second integer stored in a register.
        target (Register): register where result will be saved.
        reg_aux_1 (Register, optional): Auxiliar register. Defaults to Register().
        reg_aux_2 (Register, optional): Auxiliar register. Defaults to Register().
        reg_aux_3 (Register, optional): Auxiliar register. Defaults to Register().

    Returns:
        Block: Block containing three blocks with the Q1ASM program needed
    """
    # Block creation
    product = Block(f"multiply_{multiply.calls}")
    # Q1ASM program
    # init
    init = Block(f"multiply_init_{multiply.calls}")
    init.append_components(
        [
            Move(var=a, register=reg_aux_1),  # multiply-by-two variable
            Move(var=b, register=reg_aux_2),  # divide-by-two variable
            Move(var=0, register=target),  # product result
        ]
    )
    # loop
    loop = Block(f"multiply_loop_{multiply.calls}")
    loop.append_components(
        [
            And(origin=reg_aux_2, var=1, destination=reg_aux_3),  # Exctract less significant bit
            Nop(),  # Wait before reading a register that was writen just before
            Jlt(a=reg_aux_3, b=1, instr=f"@multiply_skip_add_{multiply.calls}"),  # If is 0 jump to skip_add
            Add(origin=target, var=reg_aux_1, destination=target),  # Sum
        ]
    )
    # skip_add
    skip_add = Block(f"multiply_skip_add_{multiply.calls}")
    skip_add.append_components(
        [
            Asr(origin=reg_aux_2, var=1, destination=reg_aux_2),  # Divide by two
            Asl(origin=reg_aux_1, var=1, destination=reg_aux_1),  # Multiply by two
            Jge(a=reg_aux_2, b=1, instr=f"@multiply_loop_{multiply.calls}"),  # Jump to loop if not finished
        ]
    )
    # Ensamble all blocks
    product.append_components([init, loop, skip_add])
    return product
