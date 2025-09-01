from qibo import gates

def hartree_to_mj_per_mol(energy_hartree: float) -> float:
    """
    From Hartree to MJ/mol.
    """
    return energy_hartree * 2.62549962

def rotate_to_z(circuit, pauli_str):
    nqubits = circuit.nqubits

    # Expandir string de Pauli a longitud nqubits rellenando con I
    full_pauli = ['I'] * nqubits
    for term in pauli_str:
        if len(term) == 0:   # identidad
            continue
        op = term[0]
        q = int(term[1:])
        full_pauli[q] = op

    """
    Rotate qubits in a Qibo circuit so that X and Y operators can be measured in Z basis.

    Args:
        circuit: Qibo Circuit object
        pauli_str: string of Pauli operators, e.g. 'IXYZ'

    Returns:
        circuit: modified Qibo circuit with rotations applied
    """
    for q, op in enumerate(full_pauli[::-1]):
        if op == 'X':
            circuit.add(gates.H(q))       # X → Z
        elif op == 'Y':
            circuit.add(gates.SDG(q))     # Y → Z: apply S†
            circuit.add(gates.H(q))       # then H
        # Z and I do not need rotation
    return circuit

def rotate_to_zmod(circuit, pauli_term, nqubits):
    """
    Applies basis-change rotations so that measurements of Pauli X or Y operators
    can be performed in the computational Z-basis. Identity operators are handled
    correctly.

    Args:
        circuit: Qibo Circuit object
        pauli_term: (list[str] or None): A list of Pauli operators in string form, each element specifies the operator (X, Y, Z, I)
            and the target qubit index. If None or empty, the identity is assumed.
        nqubits (int): Total number of qubits in the circuit.

    Returns:
        circuit: The same Qibo Circuit object with the appropriate rotation
        gates added to map the given Pauli operators into the Z-basis.
    """
    full_pauli = ['I'] * nqubits
    if pauli_term is not None:
        for op_str in pauli_term:
            if not op_str:  # identity
                continue
            op = op_str[0]           
            q = int(op_str[1:])      
            full_pauli[q] = op

    for q, op in enumerate(full_pauli):
        if op == 'X':
            circuit.add(gates.H(q))
        elif op == 'Y':
            circuit.add(gates.SDG(q))
            circuit.add(gates.H(q))
    return circuit

from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z
from openfermion import QubitOperator


def op_to_qibo(hamiltonian, nqubits):
    """
    Turns a hamiltonian from OpenFermion to a SymbolicHamiltonian of Qibo.
    
    Args:
        hamiltonian (QubitOperator or list): it can be a QubitOperator of OpenFermion
                                             or a list of (coef + term_list).
        nqubits (int): Total number of qubits in the circuit.
        
    Returns:
        SymbolicHamiltonian: Hamiltonian in Qibo format.
    """
    # If it is a QubitOperator, rewrite it a list of (coef, term_list)
    if isinstance(hamiltonian, QubitOperator):
        h_list = []
        for term, coef in hamiltonian.terms.items():
            term_list = [f"{p}{q}" for q, p in term]
            h_list.append((coef, term_list))
    else:
        h_list = hamiltonian

    # Build SymbolicHamiltonian
    sym_h = 0
    SYMBOLS = {"X": X, "Y": Y, "Z": Z, "I": I}

    for coef, term in h_list:
        if not term:  # identidad
            sym_h += coef * I(0)
        else:
            term_sym = 1
            for op_str in term:
                pauli, qubit = op_str[0], int(op_str[1:])
                term_sym *= SYMBOLS[pauli](qubit)
            sym_h += coef * term_sym

    return SymbolicHamiltonian(sym_h)

def print_energies(energies):
    """
    Prints the energy values stored in a nested dictionary, grouped by method
    and unit.

    Args:
        energies (dict): A dictionar of methods plus their units.
    Returns:
        None: This function only prints the formatted results.
    """
    for method, vals in energies.items():
        print(f"{method}:")
        for unit, val in vals.items():
            print(f"  {unit}: {val}")