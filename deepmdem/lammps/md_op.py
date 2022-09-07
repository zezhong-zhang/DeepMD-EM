from pathlib import Path
from os.path import basename
from dflow.python import OP, OPIO, Artifact, OPIOSign
from typing import List

# import random
import os
import subprocess
import dpdata
from deepmdem.lammps.periodic_table import elements_dict


class MDRelax(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "pbc": bool,
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "running_cores": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),
                "out_structure": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            atomic_potential=basename(op_in["atomic_potential"]),
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        system = dpdata.System(
            "relax.dump",
            fmt="lammps/dump",
            type_map=list(op_in["element_map"].keys()),
            unwrap=True,
        )
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        op_out = OPIO(
            {
                "out_art": [Path("input.lammps"), Path("log"), Path("relax.dump")],
                "out_structure": Path("output_structure.lammps"),
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
        ret += "min_style cg\n"
        ret += "minimize 1e-15 1e-15 10000 10000\n"
        ret += "undump 1\n"
        return ret


class MDHeating(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "temp": float,
                "nsteps": int,
                "ensemble": str,
                "pbc": bool,
                "pres": float,
                "timestep": float,
                "neidelay": int,
                "trj_freq": int,
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "running_cores": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),
                "out_structure": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            ensemble=op_in["ensemble"],
            atomic_potential=basename(op_in["atomic_potential"]),
            nsteps=op_in["nsteps"],
            timestep=op_in["timestep"],
            neidelay=op_in["neidelay"],
            trj_freq=op_in["trj_freq"],
            temp=op_in["temp"],
            pres=op_in["pres"],
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        system = dpdata.System(
            "heating.dump",
            fmt="lammps/dump",
            type_map=list(op_in["element_map"].keys()),
            unwrap=True,
        )
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        op_out = OPIO(
            {
                "out_art": [Path("input.lammps"), Path("log"), Path("heating.dump")],
                "out_structure": Path("output_structure.lammps"),
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        ensemble: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        pres: float = 0.0,
        tau_t: float = 0.5,
        tau_p: float = 0.5,
        pbc: bool = True,
    ):
        if "npt" in ensemble and pres is None:
            raise RuntimeError("the pressre should be provided for npt ensemble")
        ret = "variable        NSTEPS          equal %d\n" % nsteps
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "variable        TEMP            equal %f\n" % temp
        if timestep is not None:
            ret += "variable        TIMESTEP        equal %f\n" % timestep
        if pres is not None:
            ret += "variable        PRES            equal %f\n" % pres
        ret += "variable        TAU_T           equal %f\n" % tau_t
        if pres is not None:
            ret += "variable        TAU_P           equal %f\n" % tau_p
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

        ret += "timestep ${TIMESTEP}\n"
        ret += "velocity all create 10 825577 dist gaussian\n"
        if pbc is True:
            ret += "fix heating all npt temp 10 ${TEMP} ${TAU_P} iso ${PRES} ${PRES} ${PRES}\n"
        else:
            ret += "fix heating all nvt temp 10 ${TEMP} ${TAU_T}\n"
            ret += "velocity all zero angular\n"
            ret += "velocity all zero linear\n"
        ret += "dump 1 all custom ${DUMP_FREQ} heating.dump id type xu yu zu\n"
        ret += "run %d\n" % nsteps
        ret += "unfix heating\n"
        ret += "undump 1\n"
        return ret


class MDStabilization(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "temp": float,
                "pres": float,
                "nsteps": int,
                "pbc": bool,
                "timestep": float,
                "neidelay": int,
                "trj_freq": int,
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "add_vacuum": float,
                "running_cores": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),
                "out_structure": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            atomic_potential=basename(op_in["atomic_potential"]),
            nsteps=op_in["nsteps"],
            timestep=op_in["timestep"],
            neidelay=op_in["neidelay"],
            trj_freq=op_in["trj_freq"],
            temp=op_in["temp"],
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
            pres=op_in["pres"],
            add_vacuum=op_in["add_vacuum"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        system = dpdata.System(
            "stabilization.dump",
            fmt="lammps/dump",
            type_map=list(op_in["element_map"].keys()),
            unwrap=True,
        )
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        op_out = OPIO(
            {
                "out_art": [
                    Path("input.lammps"),
                    Path("log"),
                    Path("stabilization.dump"),
                ],
                "out_structure": Path("output_structure.lammps"),
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        tau_t: float = 0.5,
        tau_p: float = 0.5,
        pbc: bool = True,
        pres: float = None,
        add_vacuum: float = 0.0,
    ):
        ret = "variable        NSTEPS          equal %d\n" % nsteps
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "variable        TEMP            equal %f\n" % temp
        if pres is not None:
            ret += "variable        PRES            equal %f\n" % pres
        ret += "variable        TIMESTEP        equal %f\n" % timestep
        ret += "variable        TAU_T           equal %f\n" % tau_t
        if pres is not None:
            ret += "variable        TAU_P           equal %f\n" % tau_p
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

        ret += "timestep ${TIMESTEP}\n"

        if pbc:
            ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
            ret += "fix stable_npt all npt temp ${TEMP} ${TEMP} ${TAU_P} iso ${PRES} ${PRES} ${PRES}\n"
            ret += "run %d\n" % nsteps
            ret += "unfix stable_npt\n"
            ret += "min_style cg\n"
            ret += "minimize 1e-15 1e-15 10000 10000\n"
            if add_vacuum > 0.0:
                ret += (
                    "change_box all z delta -{vacuum} {vacuum} boundary p p f\n".format(
                        vacuum=add_vacuum / 2
                    )
                )
            ret += "fix stable_nvt all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
            ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
            ret += "velocity all zero angular\n"
            ret += "velocity all zero linear\n"
            ret += (
                "dump 1 all custom ${DUMP_FREQ} stabilization.dump id type xu yu zu\n"
            )
            ret += "run %d\n" % nsteps
            ret += "unfix stable_nvt\n"
            ret += "undump 1\n"
        else:
            ret += "fix stable_nve all nve\n"
            ret += "fix rescalet all temp/rescale 10 ${TEMP} ${TEMP} 30.0 1.0\n"
            ret += (
                "dump 1 all custom ${DUMP_FREQ} stabilization.dump id type xu yu zu\n"
            )
            ret += "run %d\n" % nsteps
            ret += "unfix stable_nve\n"
            ret += "unfix rescalet\n"
            ret += "undump 1\n"
            ret += "\n"
        return ret


class MDEquilibrium(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "temp": float,
                "nsteps": int,
                "pbc": bool,
                "timestep": float,
                "neidelay": int,
                "trj_freq": int,
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "running_cores": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({"out_art": Artifact(List[Path])})

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            atomic_potential=basename(op_in["atomic_potential"]),
            nsteps=op_in["nsteps"],
            timestep=op_in["timestep"],
            neidelay=op_in["neidelay"],
            trj_freq=op_in["trj_freq"],
            temp=op_in["temp"],
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        op_out = OPIO(
            {"out_art": [Path("input.lammps"), Path("log"), Path("equilibrium.dump")]}
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        tau_t: float = 0.5,
        pbc: bool = True,
    ):
        ret = "variable        NSTEPS          equal %d\n" % nsteps
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "variable        TEMP            equal %f\n" % temp
        if timestep is not None:
            ret += "variable        TIMESTEP        equal %f\n" % timestep
        ret += "variable        TAU_T           equal %f\n" % tau_t
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
        ret += "timestep ${TIMESTEP}\n"
        ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
        ret += "fix equilibrium all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
        if not pbc:
            ret += "velocity all zero angular\n"
            ret += "velocity all zero linear\n"

        ret += "dump 1 all custom ${DUMP_FREQ} equilibrium.dump id type xu yu zu\n"
        ret += "run %d\n" % nsteps
        ret += "unfix equilibrium\n"
        ret += "undump 1\n"
        return ret
