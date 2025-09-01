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

# Prefix for a Q1ASM register identifier (prefix + number).
PROG_REG_PREFIX = "R"
# Maximum number of registers in a program
PROG_MAX_REGISTERS = 64
# Minimum wait in nanoseconds for an instruction with a wait parameter
INST_MIN_WAIT = 4
# Maximum wait in nanoseconds for an instruction with a wait parameter
INST_MAX_WAIT = 2**16 - 4
# Maximum value of the NCO phase (equivalent to 360º).
NCO_MAX_INT_PHASE = int(1e9)
# Minimum int representation value of the NCO frequency (used by Q1ASM instructions).
NCO_MIN_INT_FREQ = int(-2e9)
# Maximum int representation value of the NCO frequency (used by Q1ASM instructions).
NCO_MAX_INT_FREQ = int(2e9)
# Minimum value the NCO frequency in Hz.
NCO_MIN_HZ_FREQ = int(-0.5e9)
# Maximum value the NCO frequency in Hz.
NCO_MAX_HZ_FREQ = int(0.5e9)
# NCO frequency Hz to int multiplier.
NCO_HZ_TO_INT_MULTIPLIER = 4
# Minimum gain of the Arbitrary Waveform Generator
AWG_MIN_GAIN = -(2**15)
# Maximum gain of the Arbitrary Waveform Generator
AWG_MAX_GAIN = 2**15 - 1
# Minimum offset of the Arbitrary Waveform Generator
AWG_MIN_OFFSET = -(2**15)
# Maximum offset of the Arbitrary Waveform Generator
AWG_MAX_OFFSET = 2**15 - 1
# Minimum value for an immediate argument type
IMMD_MIN_VALUE = -(2**32)
# Maximum value for an immediate argument type
IMMD_MAX_VALUE = 2**32 - 1
# Number of markers in a sequencer
SEQ_N_MARKERS = 4
# Step of waiting times for sequencer instructions
SEQ_WAIT_STEP = 4
# Maximum value of a mask used in latch instructions
MASK_MAX_VALUE = 2**15 - 1
