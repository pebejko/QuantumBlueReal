from openfermionpyscf import run_pyscf
from src.utils import hartree_to_mj_per_mol
from openfermion import MolecularData

def build_molecule(config):
    '''
    Construct  the MolecularData object and run HF and FCI according to the configuration.
    Returns mol_data with reference energies and properties.
    '''

    # Read the geometry and parameters to build the system from the config file:
    geometry = config["geometry"] 
    basis = config.get("basis", "sto-3g")
    multiplicity = config.get("multiplicity", 1)
    charge = config.get("charge", 0)

    mol = MolecularData(geometry=geometry, 
                        basis=basis,
                        multiplicity=multiplicity, 
                        charge=charge)
    
    mol_data = run_pyscf(molecule=mol,
                         run_scf=True,
                         run_fci=True)
    return mol_data

def get_reference_energies(mol_data):
    """
    Returns HF and FCI energies in Hartree and kJ/mol.
    """
    hf_energy_hartree = mol_data.hf_energy
    fci_energy_hartree = mol_data.fci_energy
    
    hf_energy_mj = hartree_to_mj_per_mol(hf_energy_hartree)
    fci_energy_mj = hartree_to_mj_per_mol(fci_energy_hartree)
    
    return {
        "HF": {"Ha": hf_energy_hartree, "MJ/mol": hf_energy_mj},
        "FCI": {"Ha": float(fci_energy_hartree), "MJ/mol": float(fci_energy_mj)}
    }

