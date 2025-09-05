import torch
import torch.optim as optim
import torch.nn as nn
import os
import pickle
import yaml
import numpy as np
#import qililab as ql
from openfermion.utils import count_qubits
from src.hamiltonian import build_pauli_hamiltonian_qibo
from src.utils import rotate_to_zmod
from src.molecule import build_molecule
from qibo import gates 
from collections import Counter

class VQERunner:
    """
    A Variational Quantum Eigensolver (VQE) runner that integrates
    a quantum ansatz with a classical PyTorch model and optimizes parameters
    using Adam optimizer.

    This class manages:
      - Configuration loading from a YAML file
      - Backend setup (simulator or hardware)
      - Molecule construction and Hamiltonian generation
      - Ansatz integration (circuit builder)
      - Energy estimation via Pauli measurements
      - Gradient computation with parameter-shift
      - Training loop with checkpoints

    Args:
        config_path (str): Path to the YAML configuration file. Must contain
            backend setup, optimizer parameters, molecule/ansatz info and checkpoint info.
        model (nn.Module): A PyTorch model containing the trainable parameters
            for the quantum ansatz.
        ansatz: An ansatz object with attributes `nlayers`, `connectivity`, and
            a `build_circuit(params)` method returning a Qibo Circuit.

    Attributes:
        config (dict): Parsed configuration dictionary from YAML.
        checkpoint_file (str): Path for saving checkpoints.
        backend_path (str): Simulator or hardware backend name.
        runcard (dict or None): Hardware execution config if required.
        nshots (int): Number of measurement shots per circuit execution.
        seed (int or None): Random seed for reproducibility.
        mol_data: Molecular data built from config.
        hamiltonian_op: Qibo symbolic Hamiltonian operator.
        hamiltonian (list): List of (coefficient, Pauli-term) tuples.
        nqubits (int): Number of qubits required by the Hamiltonian.
        ansatz: Stored ansatz object.
        model (nn.Module): PyTorch model with variational parameters.
        optimizer (torch.optim.Optimizer): Adam optimizer instance.
        max_epochs (int): Number of training iterations.

    Methods:
        save_checkpoint(params, energies, step):
            Saves model parameters and energies at a given iteration.
        load_checkpoint():
            Loads a previous checkpoint if available.
        measure_hamiltonian(circuit, shots=None):
            Estimates the expectation value of the Hamiltonian for a circuit.
        run():
            Executes the full VQE optimization loop and returns energy history.
    """

    def __init__(self, config_path: str, model: nn.Module, ansatz):
        with open(config_path, "r") as f: # Load configuration from YAML (backend, optimizer, molecule, VQE options)
            self.config = yaml.safe_load(f)

        # Define checkpoint file (used to resume training)
        self.checkpoint_file = self.config.get("checkpoint_file", "results/checkpoint.pkl")

        # Backend setup: simulator (default qibojit) or real quantum hardware
        backend_cfg = self.config.get("backend", {})
        sim_backend = backend_cfg.get("simulator", "qibojit")
        self.backend_path = sim_backend if isinstance(sim_backend, str) else sim_backend.get("name", "qibojit")
        self.runcard = backend_cfg.get("runcard", None)
        if self.backend_path != "qibojit" and self.runcard is None:
            raise ValueError("Para backend de hardware, debes especificar 'runcard' en el config.")

        # Total number of shots
        self.nshots = backend_cfg.get("shots", 500)

        # Set random seed for reproducibility if none is declared
        self.seed = self.config.get("seed", None)
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Build molecular Hamiltonian
        self.mol_data = build_molecule(self.config)
        self.hamiltonian_op, pauli_terms = build_pauli_hamiltonian_qibo(self.mol_data, self.config)
        self.hamiltonian = pauli_terms  # list of coef + terms
        self.nqubits = count_qubits(self.hamiltonian_op)

        # Store ansatz description (layers, connectivity, circuit builder).
        self.ansatz = ansatz
        self.nlayers = ansatz.nlayers
        self.connectivity = ansatz.connectivity

        # Initialize PyTorch optimizer (Adam).
        self.model = model
        adam_defaults = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
        adam_params_raw = self.config.get("adam_params", {})
        adam_params = {**adam_defaults, **adam_params_raw} # Override defaults params with config params for Adam.
        for key in ["lr", "eps", "weight_decay"]: # Make sure numerical types are floats.
            if key in adam_params: adam_params[key] = float(adam_params[key])
        if "betas" in adam_params:
            adam_params["betas"] = tuple(float(b) for b in adam_params["betas"])

        self.optimizer = optim.Adam(self.model.parameters(), **adam_params)

        # VQE options: restart, checkpoint interval, print interval and maximum number of epochs.
        self.max_epochs = self.config.get("max_epochs", 100)
        vqe_cfg = self.config.get("vqe", {})
        self.restart = vqe_cfg.get("restart", False)  
        self.checkpoint_interval = vqe_cfg.get("checkpoint_interval", 100) 
        self.print_interval = vqe_cfg.get("print_interval", 1)

        print(f"VQERunner inicializado: Backend={self.backend_path}, nqubits={self.nqubits}, Adam={adam_params}, checkpoint_interval={self.checkpoint_interval}, "
              f"print_interval={self.print_interval}")

    def save_checkpoint(self, params, energies, step): # Save model parameters, energy history, and current iteration for restart.
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump({"params": params, "energies": energies, "step": step}, f)
        print(f"[Checkpoint] Iteración {step} guardada")

    def load_checkpoint(self): # Resume from last checkpoint if available, otherwise start fresh.
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "rb") as f:
                data = pickle.load(f)
            print(f"[Checkpoint] Reanudando desde iteración {data['step']}")
            return data["params"], data["energies"], data["step"]
        return None, [], 0

    def _expectation_from_samples(self, freq, term): # Compute expectation value of a Pauli term from sampled bitstring frequencies (assumes measurement in Z basis after basis-change rotations).
        nshots = sum(freq.values()) # Number of total measurments
        if not term: 
            return 1.0 # If the term is the identity, his expected value is always 1.
        expval = 0.0
        for bitstring, count in freq.items(): # For each operator of the term, check whether the corresponding qubit is in state 1. First settig parity to 1.
            parity = 1
            for op_str in term:
                if not op_str: continue
                q = int(op_str[1:])
                if bitstring[self.nqubits - q - 1] == '1':
                    parity *= -1 # If the qubit is in state 1, multiplies parity by -1 (applying +/- rule for Pauli Z).
            expval += parity * count / nshots # Add the weighted value for how many times that bitstring appeared, and then return the value. 
        return expval

    def measure_hamiltonian(self, circuit, shots=None): # Estimate total hamiltonian expectation value by measuring each Pauli term
        if shots is None:
            shots = self.nshots # If number of shots is not defined, use the number of measurements.
        total_energy = 0.0 
        for coef, term in self.hamiltonian: # After setting energy to 0, run through each term of the hamiltonian.
            circ_copy = circuit.copy()
            circ_copy = rotate_to_zmod(circ_copy, term, self.nqubits)
            circ_copy.add(gates.M(*range(self.nqubits))) # Creates a copy of the original circuit, apllies rotations to basis Z, and then measures all the qubits.

            if self.backend_path== "qibojit": # Executes circuit in backend. Simulator (qibojit) or real quantum computer (runcard)
                result = circ_copy(nshots=shots)
            else:
                #result = ql.execute(circ_copy, self.runcard, nshots = shots)
                result = circ_copy(nshots=shots)
            freq = Counter(result.frequencies()) # How many times each bitstring appeared (freq)
            term_energy = coef * self._expectation_from_samples(freq, term) # Calculat the energy of this term, using _expectation_from_samples().
            total_energy += term_energy # Adds the energy of every term caculated before to get the total energy.

        return float(total_energy)

    def _energy_and_gradients(self, params_np, shift=np.pi/2): # Compute energy and parameter gradients via parameter-shift rule
        n_params = len(params_np)
        grads = np.zeros_like(params_np) # Empty array to save all the gradients
        circuit = self.ansatz.build_circuit(params_np) # Build the the circuit qith the current parameters
        energy = self.measure_hamiltonian(circuit) # Measures the energy with that parameters

        for i in range(n_params): # For each paraemter, create 2 versions of the circuit:
            params_plus = params_np.copy() # one with the parameter increased by a shift. 
            params_minus = params_np.copy() # the other decreased by the same shift. 
            params_plus[i] += shift
            params_minus[i] -= shift
            circ_plus = self.ansatz.build_circuit(params_plus)
            circ_minus = self.ansatz.build_circuit(params_minus)
            E_plus = self.measure_hamiltonian(circ_plus) # Measure the energy for each case
            E_minus = self.measure_hamiltonian(circ_minus)
            grads[i] = 0.5 * (E_plus - E_minus) # Calculate the derivative of the energy with respect to that parameter using the parameter-shift rule.

        return energy, grads 

    def run(self):
        """
        Main VQE loop: evaluate energy, compute gradients, update parameters.

        Supports restart from checkpoint and configurable logging/checkpoint intervals.
        """

        restart = self.config.get("vqe", {}).get("restart", False) # Restart or checkpoint iniciation 
        if restart:
            energies = []
            start_iter = 0
            print("[VQE] Forced restart")
        else:
            params, energies, start_iter = self.load_checkpoint()
            if params is not None:
                self.model.load_state_dict(params)

        for epoch in range(start_iter, self.max_epochs): # Iter from the beggining up to the máximum number of epochs (in our case epoch = iteration).
            params_tensor = torch.nn.utils.parameters_to_vector(self.model.parameters()) # Turn all the parameters of the Pytorch model into a numpy array.
            params_np = params_tensor.detach().numpy()

            energy, grads_np = self._energy_and_gradients(params_np) # Calculate energy and grads of the actual circuit using _energy_and_gradients(). 

            grads_tensor = torch.tensor(grads_np, dtype=torch.float32) # Turns the grads into a Pytorch tensor, necessary to assign them to the model and use Adam.

            self.optimizer.zero_grad() # Erase grads to update parms.

            offset = 0
            for p in self.model.parameters(): # Loop over each parameter in the model to assign the corresponding grad.
                numel = p.numel() # Count how many elements (numbers) this parameter has.
                p.grad = grads_tensor[offset:offset+numel].view_as(p)
                offset += numel # Move the offset forward for the next parameter

            self.optimizer.step()
            energies.append(energy) # Update parameters using Adam optimizer, and save the energy of the epoch into a list of energies.

            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(self.model.state_dict(), energies, epoch) # Every checkpoint_interval epochs, save a checkpoint with the current parameters and energies.
            if epoch % self.print_interval == 0:
                print(f"Epoch {epoch}: Energy = {energy:.6f}") # Every print_interval epochs, print the current energy to monitor progress.

        return energies