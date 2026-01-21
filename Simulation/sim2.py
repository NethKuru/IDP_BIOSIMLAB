#!/usr/bin/env python

import openmm.app as app
import openmm as mm
import openmm.unit as u
import numpy as np
import sys
import os
import numpy as np
import time

# --- 1. Simulation Parameters ---
backbone_radius = 1.9
bead_type_hash = {}
bead_type_hash["G"] = 2.25
bead_type_hash["A"] = 2.52
bead_type_hash["R"] = 3.28
bead_type_hash["K"] = 3.18
bead_type_hash["H"] = 3.04
bead_type_hash["D"] = 2.79
bead_type_hash["E"] = 2.96
bead_type_hash["S"] = 2.59
bead_type_hash["T"] = 2.81
bead_type_hash["N"] = 3.01
bead_type_hash["C"] = 2.74
bead_type_hash["P"] = 2.78
bead_type_hash["I"] = 3.09
bead_type_hash["M"] = 3.09
bead_type_hash["L"] = 3.09
bead_type_hash["F"] = 3.18
bead_type_hash["W"] = 3.39
bead_type_hash["Y"] = 3.23
bead_type_hash["V"] = 2.93
# Input/Output Files
PDB_FILE = 'sys_final.pdb'
OUTPUT_DCD = 'out.dcd'
OUTPUT_LOG = 'out.log'
OUTPUT_FINAL_PDB = 'sys_final.pdb'
BOND_LIST = "bond_dist.dat"

# Simulation Control
TOTAL_STEPS = 2     # Total steps to run
REPORT_INTERVAL = 1    # How often to save frames and log data

# Simulation Conditions
TEMPERATURE = 300 * u.kelvin
FRICTION_COEF = 1.0 / u.picoseconds
TIME_STEP = 2.0 * u.femtoseconds

# --- 2. FENE Bond Parameters ---

unit_len = 0.1 * u.nanometer



print("--- FENE  & EXCLUDED VOLUME FORCE Bond Simulation ---")
print(f"Loading PDB: {PDB_FILE}...")
pdb = app.PDBFile(PDB_FILE)
    
positions = pdb.positions
topology = pdb.topology

# --- 3. Create the OpenMM System ---
system = mm.System()
box = 300 * u.angstrom
system.setDefaultPeriodicBoxVectors([box,0,0], [0,box,0], [0,0,box])

# Add particles to the system, inferring mass from the PDB's element info
print("Adding particles to system...")
# -----------------------FIX MASSES-------------------------------------#

for atom in topology.atoms():
    if atom.name == 'CA':
        system.addParticle(55.0 * u.amu)
    elif atom.name == 'SC':
        system.addParticle(55.0 * u.amu)
    else:
        system.addParticle(55.0 * u.amu) 




        
# k_bond: Bond stiffness
k_val = 83.68
k_bond = k_val * u.kilojoules_per_mole / (unit_len**2)


R0_val = 2      
R0 = R0_val * unit_len

print(f"Loading bonds from {BOND_LIST}...")

bondList = np.loadtxt(BOND_LIST, unpack=True)

print("Adding FENE bond force...")

NBond = len(bondList[0])

fene_formula = "-0.5 * k_fene*R0^2*log(1-((r-r0)^2.0)/R0^2.0)"

feneBond = mm.CustomBondForce(fene_formula)

# Define the parameters
feneBond.addPerBondParameter('k_fene') 
feneBond.addPerBondParameter('R0')     
feneBond.addPerBondParameter('r0')  
num_bonds_added = 0
for i in range(len(bondList[0])):
    idx1 = int(bondList[0][i])
    idx2 = int(bondList[1][i])
    r0_dist = float(bondList[2][i])
#     print(idx1,idx2,r0_dist)

    feneBond.addBond(idx1, idx2, [k_bond, R0, r0_dist * unit_len])
    num_bonds_added += 1

system.addForce(feneBond)
#------------------------Adding EXCLUDED VOLUME FORCE -----------------------------#

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
  


excludedVolume_formula = "epsilonEV * (sigma/r)^6"
excludedVolume = mm.CustomBondForce(excludedVolume_formula)
#Define Parameters
#epsilon = 1 kcal/mol =  4.184 kj/mol

epsilon = 4.184 * u.kilojoules_per_mole
excludedVolume.addPerBondParameter('epsilonEV')
excludedVolume.addPerBondParameter('sigma')
#Value of parameters
count_ev = 0

for i in range(0, NOM - 2, 2):
    for j in range(i + 3, i + 6):
        if j < NOM:
            sigma_NL = (sigma_list[i] + sigma_list[j]) * unit_len
            excludedVolume.addBond(i, j, [epsilon, sigma_NL])
            print(i, j, epsilon, sigma_NL,sigma_list[i],sigma_list[j])
            count_ev += 1


for i in range(1, NOM - 1, 2):
    for j in range(i + 1, i + 5):
        if j < NOM:
            sigma_NL = (sigma_list[i] + sigma_list[j]) * unit_len
            excludedVolume.addBond(i, j, [epsilon, sigma_NL])
            count_ev += 1
#system.addForce(excludedVolume)
print(f"Excluded Volume force added with {excludedVolume.getNumBonds()} pairs.")

#----------------------------------------------screened Coulomb potential------------------#


# Use a Langevin integrator for NVT simulation
# --- Set up the Simu ---
print("Setting up integrator and simulation...")

# Langevin integrator
integrator = mm.LangevinIntegrator(TEMPERATURE, FRICTION_COEF, TIME_STEP)

# Try to use the CUDA platform for speed, fall back to CPU
try:
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    print("Running on CUDA platform.")
except mm.OpenMMException:
    platform = mm.Platform.getPlatformByName('CPU')
    simulation = app.Simulation(topology, system, integrator, platform)
    print("CUDA not found. Running on CPU platform.")

# Set the initial positions from the PDB
simulation.context.setPositions(positions)

# --- Minimize ---
print("Minimizing energy...")
simulation.minimizeEnergy()

# Set initial velocities to match the simulation temperature
simulation.context.setVelocitiesToTemperature(TEMPERATURE)

# --- 8. Configure Reporters ---
print("Setting up reporters...")
os.chdir(f"/user/nethkuru/shared/Neth/IDP_BIOSIMLAB/Simulation/test_runs")
base_dir = 'FENE_EV_run_'
run_num = 0
while True:
    output_dir = f"{base_dir}{run_num}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new directory: {output_dir}")
        os.chdir(f"/user/nethkuru/shared/Neth/IDP_BIOSIMLAB/Simulation/test_runs/{output_dir}")
        break
    run_num +=1

# Reporter for trajectory (DCD format)
simulation.reporters.append(app.DCDReporter(OUTPUT_DCD, REPORT_INTERVAL))

class EnergyReporter(object):
    def __init__ (self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval


    def __del__ (self):
        self._out.close()


    def describeNextReport(self, simulation):
        step = self._reportInterval - simulation.currentStep%self._reportInterval
        return (step, False, False, False, True)
        #return (step, position, velocity, force, energy)


    def report(self, simulation, state):
        energy = []
        self._out.write(str(simulation.currentStep))
        for i in range(system.getNumForces()):
            state = simulation.context.getState(getEnergy=True, groups={i})
            energy = state.getPotentialEnergy() / u.kilocalorie_per_mole
            self._out.write("  " + str(energy))
        self._out.write("\n")
print("\nEnergy components of initial position\n")
for i in range(system.getNumForces()):
    print(i,simulation.context.getState(getEnergy=True,groups={i}).getPotentialEnergy()) 
# Reporter for simulation data (log file and console)
state_reporter = app.StateDataReporter(
    sys.stdout,
    REPORT_INTERVAL,
    totalSteps=TOTAL_STEPS,
    step=True,
    potentialEnergy=True,
    temperature=True,
    progress=True,
    remainingTime=True,
    speed=True,
    separator='\t'
)
log_reporter = app.StateDataReporter(
    OUTPUT_LOG,
    REPORT_INTERVAL,
    totalSteps=TOTAL_STEPS,
    step=True,
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    temperature=True,
    speed=True
)
simulation.reporters.append(state_reporter)
simulation.reporters.append(log_reporter)


# --- Running the Simulation ---
print("\n--- Starting Simulation ---")
start_time = time.time()

simulation.step(TOTAL_STEPS)

end_time = time.time()
print("--- Simulation Finished ---")
print(f"Total time: {(end_time - start_time):.2f} seconds")

# --- 10. Save the Final State ---
state = simulation.context.getState(getPositions=True)
final_positions = state.getPositions()


app.PDBFile.writeFile(topology, final_positions, open(OUTPUT_FINAL_PDB, 'w'))

print(f"\nFinal positions saved to: {OUTPUT_FINAL_PDB}")
print(f"Trajectory saved to: {OUTPUT_DCD}")
print(f"Log data saved to: {OUTPUT_LOG}")
