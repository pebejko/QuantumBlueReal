# QuantumBlueReal
Variational Quantum Eigensolver implementation, aimed to be implemented in **Quantum Blue at BSC**.


---

## 📖 Overview

This project implements a **hybrid quantum-classical VQE workflow** to compute molecular ground state energies.  `config.yaml`

**Workflow Summary:**

<details>
<summary>Click to expand workflow diagram</summary>
### Content of `config.yaml`.

  
  ```
  # ------------------ Ansatz ------------------
ansatz:
  type: HEA1          # Nombre de la clase de ansatz
  layers: 2           # Número de capas
  connectivity: [[0,1]]     # Opcional: lista de tuplas de qubits a entrelazar

# ------------------ Backend ------------------
backend:
  simulator: qibojit   # "qibojit" para simulación, o "hardware" para QPU
  runcard: "/path/to/runcard"  # Solo si quieres ejecutar en QPU real
  shots: 100

# ------------------ Optimización ------------------
adam_params:
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.0

max_epochs: 150

# ------------------ Checkpoint ------------------
checkpoint_file: "results/checkpoint.pkl"

# ------------------ VQE Options ------------------
vqe:
  restart: true 
  checkpoint_interval: 50
  print_interval: 10

# Seed for reproducibility
seed: 42

# Molecular system (H2 simple)
geometry:
  - ["N", [0.0, 0.0, 0.0]]
  - ["H", [0.0, 0.0, 0.74]]   # Common distance H2
basis: "sto-3g"
multiplicity: 1
charge: 0

# Active space or frozen orbitals (opcional)
active_electrons: 3
active_orbitals: 3

# Symmetry options
use_symmetry: "true"
```

</details>
