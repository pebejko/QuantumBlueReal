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

from bisect import insort
from dataclasses import dataclass

import numpy as np


@dataclass
class Waveform:
    """Waveform dataclass"""

    index: int
    name: str
    data: list[float] | np.ndarray


@dataclass
class WaveformPair:
    """WaveformPair dataclass"""

    waveform_i: Waveform
    waveform_q: Waveform
    name: str


class Waveforms:
    """Waveforms class. Holds all the waveforms used in a sequence."""

    def __init__(self):
        self._used_indices: list[int] = []
        self._waveforms: list[Waveform] = []
        self._waveform_pairs: list[WaveformPair] = []

    def _find_available_index(self) -> int:
        """Finds the smallest available index

        Returns:
            index (int): available index.
        """
        index = -1
        for proposed, existing in enumerate(self._used_indices):
            if proposed != existing:
                index = proposed
        if index == -1:
            index = len(self._used_indices)
        return index

    def add(self, array: list[float] | np.ndarray, index: int = -1, name: str | None = None) -> int:
        """Adds a waveform to the Waveforms object.

        Args:
            array (list[float] | np.ndarray): array containing the waveform.
            index (int, optional): index of the waveform. Defaults to None.
            name (str, optional): string identifier of the waveform. Defaults to -1.

        Returns:
            index (int): index of the waveform.
        """
        # Check if the index is valid or find one if not given.
        if index == -1:
            index = self._find_available_index()
        elif index in self._used_indices:
            raise IndexError(f"Index {index} is already used.")
        # Add the index to the used_indices list
        insort(self._used_indices, index)
        # Auto-name if name not given.
        if not name:
            name = f"waveform_{index}"
        # If array is a numpy array, convert to list
        if isinstance(array, np.ndarray):
            array = array.tolist()  # type: ignore[assignment]
        # Create waveform object and add to waveforms list
        waveform = Waveform(index, name, array)
        self._waveforms.append(waveform)
        return index

    def add_pair(
        self,
        pair: tuple[list[float] | np.ndarray, list[float] | np.ndarray],
        indices: tuple[int, int] = (-1, -1),
        name: str | None = None,
    ) -> tuple[int, int]:
        """Adds a waveform pair to the Waveforms object.

        Args:
            pair (tuple[list[float] | np.ndarray, list[float] | np.ndarray]): tuple with two arrays containing two
                waveforms.
            indices (tuple(int, int), optional): tuple with the indices of the waveforms. Defaults to None.
            name (str, optional): string identifier of the waveform. Defaults to None.

        Returns:
            indices (tuple[int, int]): indices of the waveforms.
        """
        # Auto-name if name not given
        if not name:
            name = f"pair_{len(self._waveform_pairs)}"
        i_index, q_index = indices
        i_index = self.add(pair[0], i_index, f"{name}_I")
        q_index = self.add(pair[1], q_index, f"{name}_Q")
        return self._create_waveform_pair(name, i_index, q_index)

    def add_pair_from_complex(
        self,
        array: np.ndarray,
        indices: tuple[int, int] = (-1, -1),
        name: str | None = None,
    ) -> tuple[int, int]:
        """Adds a the real and imaginary part of a waveform as a waveform pair to the Waveforms object.

        Args:
            array (np.ndarray): complex numpy array.
            indices (tuple(int, int), optional): tuple with the indices of the waveforms. Defaults to None.
            name (str, optional): string identifier of the waveform. Defaults to None.

        Returns:
            indices (tuple[int, int]): indices of the waveforms.
        """
        # Auto-name if name not given
        if not name:
            name = f"pair_{len(self._waveform_pairs)}"
        i_index, q_index = indices
        i_index = self.add(array.real, i_index, f"{name}_I")
        q_index = self.add(array.imag, q_index, f"{name}_Q")
        return self._create_waveform_pair(name, i_index, q_index)

    def modify(self, name: str, array: list[float] | np.ndarray) -> None:
        """Adds a waveform to the Waveforms object.

        Args:
            name (str): string identifier of the waveform.
            array (list[float] | np.ndarray): array containing the waveform.
        """

        waveform = self.find_by_name(name)
        if isinstance(array, np.ndarray):
            array = array.tolist()  # type: ignore[assignment]
        waveform.data = array.copy()

    def modify_pair(
        self,
        name: str,
        pair: tuple[list[float] | np.ndarray, list[float] | np.ndarray],
    ) -> None:
        """Adds a waveform pair to the Waveforms object.

        Args:
            name (str): string identifier of the waveform.
            pair (tuple[list[float] | np.ndarray, list[float] | np.ndarray]): tuple with two arrays containing two
                waveforms.
        """

        self.modify(f"{name}_I", pair[0])
        self.modify(f"{name}_Q", pair[1])

    def _create_waveform_pair(self, name: str, i_index: int, q_index: int):
        """Creates a waveform pair.

        Args:
            name (str): name of the waveform pair.
            i_index (int): index of the I component.
            q_index (int): index of the Q component.
        """
        waveforms_pair = WaveformPair(self._waveforms[-2], self._waveforms[-1], name)
        self._waveform_pairs.append(waveforms_pair)
        return i_index, q_index

    def find_by_name(self, name: str) -> Waveform:
        """Finds a Waveform by name

        Args:
            name (str)): Name of the Waveform

        Raises:
            NameError: Waveform not found

        Returns:
            Waveform: Waveform found by name.
        """
        for waveform in self._waveforms:
            if waveform.name == name:
                return waveform
        raise NameError(f"Could not find Waveform with name {name}.")

    def find_pair_by_name(self, name: str) -> WaveformPair:
        """Finds a Waveform pair by name.

        Args:
            name (str): Name of the waveform pair.

        Raises:
            NameError: Acquisition pair not found.

        Returns:
            WaveformPair: Waveform pair found by name.
        """
        for pair in self._waveform_pairs:
            if pair.name == name:
                return pair
        raise NameError(f"Could not find Waveform pair with name {name}.")

    def to_dict(self) -> dict:
        """Creates a dictionary representation of the Waveforms object. It can be passed directly to a Qblox sequencer
        when converted to string.

        Returns:
            dict (dict): dictionary representation of the waveforms.
        """
        dictionary = {}
        for waveform in self._waveforms:
            entry = {"data": waveform.data, "index": waveform.index}
            dictionary[waveform.name] = entry
        return dictionary

    def __repr__(self) -> str:
        """JSON string representation of the Waveforms obejct. It can be passed directly to a Qblox sequencer.

        Returns:
            str: JSON representation of the waveforms."""
        return str(self.to_dict())
