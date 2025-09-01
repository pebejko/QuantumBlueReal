import numpy as np
from qibo import models, gates

class HEA1:
    """
    Ansatz HEA1:
    - 3 rotations parametrized for each qubit: RX, RY, RZ.
    - Entanglement usign CNOTs based on connectivity.
    """

    def __init__(self, nqubits, nlayers, connectivity=None):
        self.nqubits = nqubits 
        self.nlayers = nlayers
        self.connectivity = connectivity
        self.n_params = nqubits * 3 * nlayers  # total number of parameters

    def build_circuit(self, params):
        """
        Build a Qibo circuit based on HEA1 structure and the parameters.
        params: array of size (n_params).
        """
        circuit = models.Circuit(self.nqubits)
        params = np.array(params).reshape((self.nlayers, self.nqubits, 3))

        for layer in range(self.nlayers):
            # Parametrized rotations for each qubit and layer
            for qubit in range(self.nqubits):
                rx, ry, rz = params[layer, qubit]
                circuit.add(gates.RX(qubit, rx))
                circuit.add(gates.RY(qubit, ry))
                circuit.add(gates.RZ(qubit, rz))

            # Entalglement 
            if self.connectivity:
                for q0, q1 in self.connectivity:
                    circuit.add(gates.CNOT(q0, q1))

        return circuit

    @classmethod
    def get_n_params(cls, nqubits, nlayers):
        """
        Automatically returns the total number of parameters in the ansatz.
        """
        return nqubits * 3 * nlayers
