#!/usr/bin/env python

import openmm.app as app
import openmm as mm
import openmm.unit as u
import numpy as np
import sys
import numpy as np
import time

# --- 1. Simulation Parameters ---

# Input/Output Files
PDB_FILE = 'system.pdb'
OUTPUT_DCD = 'out.dcd'
OUTPUT_LOG = 'out.log'
OUTPUT_FINAL_PDB = 'sys_final.pdb'
BOND_LIST = "bond_dist.dat"

# Simulation Control
TOTAL_STEPS = 20000      # Total steps to run
REPORT_INTERVAL = 500    # How often to save frames and log data

# Simulation Conditions
TEMPERATURE = 300 * u.kelvin
FRICTION_COEF = 1.0 / u.picoseconds
TIME_STEP = 2.0 * u.femtoseconds

# --- 2. FENE Bond Parameters ---

unit_len = 0.1 * u.nanometer

# k_bond: Bond stiffness
k_val = 83.68
k_bond = k_val * u.kilojoules_per_mole / (unit_len**2)


R0_val = 2      
R0 = R0_val * unit_len


print("--- FENE Bond Simulation ---")
print(f"Loading PDB: {PDB_FILE}...")
pdb = app.PDBFile(PDB_FILE)
    
positions = pdb.positions
topology = pdb.topology

# --- 3. Create the OpenMM System ---
system = mm.System()

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

# Set periodic box vectors if they exist in the PDB
if topology.getPeriodicBoxVectors():
    print("Setting periodic box vectors from PDB.")
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

print(f"Loading bonds from {BOND_LIST}...")

bondList = np.loadtxt(BOND_LIST, unpack=True)

print("Adding FENE bond force...")


fene_formula = "-0.5 * k_fene*R0^2*log(1-((r-r0)^2.0)/R0^2.0)"

# Create the CustomBondForce
feneBond = mm.CustomBondForce(fene_formula)

# Define the parameters the formula uses
feneBond.addPerBondParameter('k_fene') # Bond stiffness
feneBond.addPerBondParameter('R0')     # Max extension
feneBond.addPerBondParameter('r0')     # Equilibrium distance

num_bonds_added = 0
for i in range(len(bondList[0])):
    idx1 = int(bondList[0][i])
    idx2 = int(bondList[1][i])
    r0_dist = float(bondList[2][i])
    
    feneBond.addBond(idx1, idx2, [k_bond, R0, r0_dist * unit_len])
    num_bonds_added += 1

system.addForce(feneBond)

# Use a Langevin integrator for NVT simulation
# --- 6. Set up the Simulation ---
print("Setting up integrator and simulation...")

# Use a Langevin integrator for NVT simulation
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

# --- 7. Minimize and Equilibrate ---
print("Minimizing energy...")
simulation.minimizeEnergy()

# Set initial velocities to match the simulation temperature
simulation.context.setVelocitiesToTemperature(TEMPERATURE)

# --- 8. Configure Reporters ---
print("Setting up reporters...")

# Reporter for trajectory (DCD format)
simulation.reporters.append(app.DCDReporter(OUTPUT_DCD, REPORT_INTERVAL))

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


# --- 9. Run the Simulation ---
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