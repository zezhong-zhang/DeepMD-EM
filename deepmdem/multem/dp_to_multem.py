from enum import unique
from re import L
import numpy as np
import multem


def dp_to_multem(dp_frame, type_map, rms3d: float = 0.0):
    params = multem.CrystalParameters()
    params.na = 1
    params.nb = 1
    params.nc = 1
    params.a = dp_frame.data["cells"][0][0][0]
    params.b = dp_frame.data["cells"][0][1][1]
    params.c = dp_frame.data["cells"][0][2][2]
    lx = params.na * params.a
    ly = params.nb * params.b
    lz = params.nc * params.c

    atom_numbs = np.sum(dp_frame.data["atom_numbs"])
    rms3d = np.ones((atom_numbs, 1)) * rms3d
    occ = np.ones((atom_numbs, 1))
    region = np.zeros((atom_numbs, 1)).astype(int)
    charge = np.zeros((atom_numbs, 1)).astype(int)
    coords = dp_frame.data["coords"][0]
    # coords = coords/[lx,ly,lz]%1
    atom_types = dp_frame.data["atom_types"][:, None]
    atom_types = np.vectorize(type_map.get)(atom_types).astype(int)
    atoms = np.concatenate((atom_types, coords, rms3d, occ, region, charge), axis=1)
    atoms=atoms[atoms[:, 3].argsort()]
    atoms = multem.AtomList(totuple(atoms))
    return atoms, lx, ly, lz, params.a, params.b, params.c

def totuple(atoms):
    atom_list = []
    for atom in atoms:
        atom = [int(atom[0]),float(atom[1]),float(atom[2]),float(atom[3]), float(atom[4]), float(atom[5]), int(atom[6]), int(atom[7])]
        atom_list.append(tuple(atom))
    return atom_list