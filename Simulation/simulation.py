import openmm.app as app
import openmm as mm
import openmm.unit as u
import numpy as np

#####hardcoding bond varibles#####

r0 = 2 * u.ang #2 angstoms
k = 20 #20 kcal/A^2/mole
Temp  = 300 * u.kelvin #300 K


#####---#####
sys_pdb = app.PDBFile("system.pdb")
pos = sys_pdb.positions
print(pos)

#### Build system ##

system = mm.System()

