from pathlib import Path
from os.path import basename
from dflow.python import OP, OPIO, Artifact, OPIOSign
from typing import List

# import random
import os
import dpdata
from deepmdem.lammps.periodic_table import elements_dict
from phonolammps import Phonolammps
from phonopy import Phonopy
import dpdata
from pymatgen.core.structure import Structure


class Phonon(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "running_cores": int,
                "supercell": list,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["structure"].parent)
        input_structure = op_in["structure"]
        dpdata.System(input_structure, fmt="auto").to(
            fmt="vasp/poscar", filename="POSCAR"
        )
        Structure.from_file("POSCAR").get_primitive_structure().to(
            filename="POSCAR.primitive"
        )
        dpdata.System("POSCAR.primitive", fmt="vasp/poscar").to(
            fmt="lammps/lmp", filename="pcell.lmp"
        )
        pcell_path = Path("pcell.lmp")
        os.chdir(op_in["atomic_potential"].parent)
        ret = self.make_lmp_input(
            conf_file=pcell_path,
            atomic_potential=basename(op_in["atomic_potential"]),
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()

        phlammps = Phonolammps(
            "input.lammps",
            supercell_matrix=[
                [op_in["supercell"][0], 0, 0],
                [0, op_in["supercell"][1], 0],
                [0, 0, op_in["supercell"][2]],
            ],
            primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        unitcell = phlammps.get_unitcell()
        force_constants = phlammps.get_force_constants()
        supercell_matrix = phlammps.get_supercell_matrix()

        phonon = Phonopy(unitcell, supercell_matrix)
        phonon.force_constants = force_constants
        phonon.run_mesh(mesh=[100, 100, 100])
        phonon.auto_total_dos(filename="total_dos.dat")
        phonon.run_thermal_properties()
        phonon.auto_band_structure(write_yaml=True, filename="band.yaml")
        phonon.save(filename="phonopy_params.yaml")

        dos_plot = phonon.plot_total_DOS()
        dos_plot.savefig("total_dos.png")
        band_plot = phonon.plot_band_structure()
        band_plot.savefig("band.png")
        thermal_plot = phonon.plot_thermal_properties()
        thermal_plot.savefig("thermal.png")
        dos_band_plot = phonon.plot_band_structure_and_dos()
        dos_band_plot.savefig("dos_band.png")

        op_out = OPIO(
            {
                "out_art": [
                    Path("input.lammps"),
                    Path("log"),
                    Path("phonopy_params.yaml"),
                    Path("band.yaml"),
                    Path("total_dos.dat"),
                    Path("total_dos.png"),
                    Path("band.png"),
                    Path("thermal.png"),
                    Path("dos_band.png"),
                ],
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        element_map: dict,
        neidelay: int = 1,
        trj_freq: int = 1,
        pbc: bool = True,
    ):
        ret = "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "\n"
        ret += "units           metal\n"
        if pbc is False:
            ret += "boundary        f f f\n"
        else:
            ret += "boundary        p p p\n"
        ret += "atom_style      atomic\n"
        ret += "\n"
        ret += "neighbor        1.0 bin\n"
        if neidelay is not None:
            ret += "neigh_modify    delay %d\n" % neidelay
        ret += "\n"
        ret += "box          tilt large\n"
        ret += "read_data %s\n" % conf_file
        ret += "change_box   all triclinic\n"
        for element, value in element_map.items():
            ret += "mass            %s %f\n" % (
                int(value) + 1,
                elements_dict[element.upper()],
            )
        if potential_type == "deepmd":
            ret += "pair_style      deepmd %s \n" % atomic_potential
            ret += "pair_coeff      * *\n"
        elif potential_type == "eam":
            ret += "pair_style      eam/alloy\n"
            element_string = "".join(str(e) for e in list(element_map.keys()))
            ret += "pair_coeff      * * %s %s\n" % (atomic_potential, element_string)
        ret += "\n"
        ret += "reset_timestep	0\n"
        ret += "thermo          ${THERMO_FREQ}\n"
        ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
        ret += "dump            1 all custom ${DUMP_FREQ} relax.dump id type xu yu zu\n"
        ret += "fix 1 all box/relax x 0.0 y 0.0 z 0.0 vmax 0.001"
        ret += "min_style cg\n"
        ret += "minimize 1e-15 1e-15 10000 10000\n"
        ret += "undump 1\n"
        return ret
