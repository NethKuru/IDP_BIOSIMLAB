import openmm.app as app
import openmm as mm
import openmm.unit as u
import numpy as np

# --- Configuration (Must match your sim2.py) ---
PDB_FILE = 'system.pdb'
BOND_LIST = "bond_dist.dat"
unit_len = 0.1 * u.nanometer # 1 Angstrom
backbone_radius = 1.9
bead_type_hash = {
    "G": 2.25, "A": 2.52, "R": 3.28, "K": 3.18, "H": 3.04, "D": 2.79, 
    "E": 2.96, "S": 2.59, "T": 2.81, "N": 3.01, "C": 2.74, "P": 2.78, 
    "I": 3.09, "M": 3.09, "L": 3.09, "F": 3.18, "W": 3.39, "Y": 3.23, "V": 2.93
}

# FENE Parameters
R0_val = 2.0 
R0 = R0_val * unit_len # Max extension in nm

print(f"--- Checking {PDB_FILE} for NaN sources ---")
pdb = app.PDBFile(PDB_FILE)
positions = pdb.positions
topology = pdb.topology
atoms = list(topology.atoms())
NOM = len(atoms)

sigma_list = []
for atom in atoms:
    res_name = atom.residue.name.strip()
    atom_name = atom.name
    print(res_name)
    print(atom_name)
    if(atom_name == "CA"):
        sigma_list.append(backbone_radius)
    else: #atom_name = SC
        if res_name in bead_type_hash:
            sigma_list.append(bead_type_hash[res_name])
print(sigma_list)


def get_dist(i, j):
    pos_i = positions[i]
    pos_j = positions[j]
    diff = pos_i - pos_j
    dist_nm = np.sqrt(diff.x**2 + diff.y**2 + diff.z**2)
    return dist_nm 


# ---  Excluded Volume Overlaps ---
print("\n--- CHECKING EXCLUDED VOLUME OVERLAPS ---")



for i in range(0, NOM - 2, 2):
    for j in range(i + 3, i + 6):
        if j < NOM:
            dist = get_dist(i, j)
            sigma_nm = (sigma_list[i] + sigma_list[j]) * 0.1 
            if dist >= sigma_nm:
                print(f"Pair {i}-{j}")
                print(f"Dist: {dist}")
                print(f"Sigma: {sigma_nm}")
          


for i in range(1, NOM - 1, 2):
    for j in range(i + 1, i + 5):
        if j < NOM:
            dist = get_dist(i, j)
            sigma_nm = (sigma_list[i] + sigma_list[j]) * 0.1 
            
            if dist >= sigma_nm:
                print(f"Pair {i}-{j}")
                print(f"Dist: {dist}")
                print(f"Sigma: {sigma_nm}")
