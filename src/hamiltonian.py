from openfermion.transforms import bravyi_kitaev, freeze_orbitals,symmetry_conserving_bravyi_kitaev
from openfermion import get_fermion_operator

def build_pauli_hamiltonian_qibo(mol_data, config):         
    '''
    Construct the hamiltonian of the system in Pauli string form using OpenFermion + PySCF, and BK mapping.
    '''    
    H_mol = mol_data.get_molecular_hamiltonian() # Molecular hamiltonian
    H_ferm = get_fermion_operator(H_mol) # Fermionic hamiltonian

    # Freeze orbitals or define active space, if there is defined that way in config file:
    n_electrons = sum([mol_data.n_electrons])
    n_orbitals = mol_data.n_orbitals
    n_spin_orbitals = 2 * n_orbitals
    active_electrons = config.get("active_electrons", None)
    active_orbitals = config.get("active_orbitals", None)

    if active_electrons and active_orbitals: # Active Space, priority over freezing
        occupied_orbitals = n_electrons // 2
        n_frozen_occ = occupied_orbitals - (active_electrons // 2)
        n_frozen_virt = (n_orbitals - occupied_orbitals) - (active_orbitals - active_electrons // 2)

        occupied_indices = list(range(0, 2 * n_frozen_occ))
        unoccupied_indices = list(range(2 * (n_orbitals - n_frozen_virt), n_spin_orbitals))

        n_active_orbitals = n_spin_orbitals - len(occupied_indices) - len(unoccupied_indices)
        n_active_fermions = n_electrons - len(occupied_indices)

        H_ferm_fr = freeze_orbitals(H_ferm, occupied_indices, unoccupied_indices, prune=True)

    else: # Freeze Orbitals, applies if there's no active space defined
        occupied_indices = config.get("occupied_orbitals", [])
        unoccupied_indices = config.get("unoccupied_orbitals", [])

        if occupied_indices or unoccupied_indices:
            H_ferm_fr = freeze_orbitals(H_ferm, occupied_indices, unoccupied_indices, prune=True)

            n_active_orbitals = n_spin_orbitals - len(occupied_indices) - len(unoccupied_indices)
            n_active_fermions = n_electrons - len(occupied_indices)
    
    use_symmetry = config.get("use_symmetry", True)
    if use_symmetry:
        H_qubit = symmetry_conserving_bravyi_kitaev(
            H_ferm_fr, n_active_orbitals, n_active_fermions
        )
    else:
        H_qubit = bravyi_kitaev(H_ferm_fr)

        # Convert to Pauli strings list
    pauli_terms = []
    for term, coef in H_qubit.terms.items():
        if term == ():  # identidad
            pauli_terms.append((coef.real, []))  # usamos lista vacía en vez de 'IIII'
        else:
            term_list = []
            for qubit, op in term:
                term_list.append(f"{op}{qubit}")
            pauli_terms.append((coef.real, term_list))
    
    return H_qubit, pauli_terms