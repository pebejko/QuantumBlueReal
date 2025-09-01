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

from qpysequence.acquisitions import Acquisitions
from qpysequence.program import Program
from qpysequence.waveforms import Waveforms
from qpysequence.weights import Weights


class Sequence:
    """Sequence class. Its string representation can be obtained with the `repr()` method and be directly fed to a
    Qblox sequencer.

        Args:
            program (Program): Program of the sequence.
            waveforms (dict): Waveforms dictionary.
            acquisitions (dict): Acquisitions dictionary.
            weights (dict): Weights dictionary.
    """

    def __init__(self, program: Program, waveforms: Waveforms | None = None, acquisitions: Acquisitions | None = None, weights: Weights | None = None):
        self._program: Program = program
        self._waveforms: Waveforms = waveforms if waveforms is not None else Waveforms()
        self._acquisitions: Acquisitions = acquisitions if acquisitions is not None else Acquisitions()
        self._weights: Weights = weights if weights is not None else Weights()

    def todict(self) -> dict:
        """JSON representation of the Sequence.

        Returns:
            dict: JSON representation of the sequence.
        """
        return {
            "waveforms": self._waveforms.to_dict(),
            "weights": self._weights.to_dict(),
            "acquisitions": self._acquisitions.to_dict(),
            "program": repr(self._program),
        }

    def __repr__(self) -> str:
        """String representation of the Sequence as JSON.
        It can be converted to json and used as a direct input for the Qblox devices.

        Returns:
            str: String representation of the sequence.
        """
        return str(self.todict())
