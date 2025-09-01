import yaml
import torch
from torch import nn
from openfermion.utils import count_qubits
from src.run import VQERunner
from src.molecule import build_molecule, get_reference_energies
from src.hamiltonian import build_pauli_hamiltonian_qibo
from src.ansatz import HEA1 
from src.utils import print_energies 


# Load configuration, by reading config.yaml. 
with open("config.yaml", "r") as f: 
    config = yaml.safe_load(f)

# Build molecule and hamiltonian of the system. Also prints Qubit requirement and reference energies (HF and FCI).
mol_data = build_molecule(config)
hamiltonian_op, hamiltonian_list = build_pauli_hamiltonian_qibo(mol_data, config)
n_qubits = count_qubits(hamiltonian_op)
print("Qubit Requirement:", n_qubits)
print()
n_layers = config.get("ansatz", {}).get("layers", 1)
ref_energies = get_reference_energies(mol_data)
print("Reference Energies:")
print_energies(ref_energies)
print()

# Read ansatz configuraion from config.yaml and instantiates it with the specified number of qubits, layers, and connectivity.
ansatz_cfg = config.get("ansatz", {})
ansatz_type = ansatz_cfg.get("type", "HEA1")
connectivity = ansatz_cfg.get("connectivity", None)
ansatz_classes = {
    "HEA1": HEA1,
} # If we want to add more ansatzes, specify here. 
ansatz_cls = ansatz_classes[ansatz_type]
ansatz = ansatz_cls(n_qubits, n_layers, connectivity)

# PyTorch Model, create learnable parameters initialized near zero.
class SimpleAnsatz(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = nn.Parameter(0.01 * torch.randn(n_params))
model = SimpleAnsatz(ansatz.n_params)

# Run the VQE optimization loop.
vqe_runner = VQERunner(config_path="config.yaml", model=model, ansatz=ansatz)
energies = vqe_runner.run()
print("--- VQE terminado ---")
