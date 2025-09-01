# QuantumBlueReal
Variational Quantum Eigensolver implementation, aimed to be implemented in **Quantum Blue at BSC**.


---

## 📖 Overview

This project implements a **hybrid quantum-classical VQE workflow** to compute molecular ground state energies.  

**Workflow Summary:**

<details>
<summary>Click to expand workflow diagram</summary>
  Config.yaml
│
▼
Molecule & Hamiltonian Builder
│
▼
Ansatz Circuit (HEA1)
│
▼
Parameterized Quantum Circuit → Quantum Measurement
│
▼
Expectation Values of Hamiltonian Terms
│
▼
Gradient Computation (Parameter-Shift Rule)
│
▼
Optimizer (Adam) Updates Parameters
│
└── Loop until max_epochs
</details>
