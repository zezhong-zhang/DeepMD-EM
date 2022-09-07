from pathlib import Path
from os.path import basename
from dflow.python import OP, OPIO, Artifact, OPIOSign
from typing import List
import dpdata
import numpy as np


class PreparePhonon(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "nphonon": int,
                "skip_frame": int,
                "lammps_dump": Artifact(Path),
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
        system = dpdata.System(op_in["lammps_dump"], fmt="lammps/dump")
        idx_list = np.linspace(
            op_in["skip_frame"], len(system) - 1, op_in["nphonon"], dtype=int
        )
        file_list = []
        for idx in idx_list:
            file = f"snap_{idx}.lammps"
            file_list.append(file)
            system.to_lammps_lmp(file, frame_idx=idx)
        op_out = OPIO(
            {
                "out_art": [Path(file) for file in file_list],
            }
        )
        return op_out
