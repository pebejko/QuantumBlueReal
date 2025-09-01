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


@dataclass
class Acquisition:
    """Acquisition dataclass."""

    name: str
    num_bins: int
    index: int


class Acquisitions:
    """Acquisitions class. Holds all the acquisitions used in a sequence."""

    def __init__(self):
        self._acquisitions: list[Acquisition] = []
        self._used_indices: list[int] = []

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

    def add(self, name: str, num_bins: int = 1, index: int = -1) -> int:
        """Adds an acquisition to the Acquisitions object.

        Args:
            name (str): Name of the acquisition to add.
            num_bins (int, optional): Number of bins. Defaults to 1.
            index (int, optional): Index of de acquisition. Defaults to -1.

        Raises:
            IndexError: Index already exists.

        Returns:
            int: Index assigned to the acquisition.
        """
        # Check if the index is valid or find one if not given.
        if index == -1:
            index = self._find_available_index()
        elif index in self._used_indices:
            raise IndexError(f"Index {index} is already used.")
        # Add the index to the used indices list
        insort(self._used_indices, index)
        # Create acquisition object and add to acquisitions list
        acquisition = Acquisition(name, num_bins, index)
        self._acquisitions.append(acquisition)
        return index

    def find_by_name(self, name: str) -> Acquisition:
        """Finds an acquisition by name.

        Args:
            name (str): Name of the acquisition.

        Raises:
            NameError: Acquisition not found.

        Returns:
            Acquisition: Acquisition found by name.
        """
        for acquisition in self._acquisitions:
            if acquisition.name == name:
                return acquisition
        raise NameError(f"Could not find acquisition with name {name}.")

    def to_dict(self) -> dict:
        """Creates a dictionary representation of the Acquisition object. It can be passed directly to a Qblox
        sequencer when converted to string.

        Returns:
            dict: dictionary representation of the acquisition.
        """
        dictionary = {}
        for acquisition in self._acquisitions:
            entry = {"num_bins": acquisition.num_bins, "index": acquisition.index}
            dictionary[acquisition.name] = entry
        return dictionary

    def __repr__(self) -> str:
        """JSON string representation of the Acquisitions object. It can be directly passed to a Qblox sequencer when
        converted to string sequencer.

        Returns:
            str: JSON representation of the acquisitions.
        """
        return str(self.to_dict())
