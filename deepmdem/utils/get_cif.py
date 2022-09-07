from pathlib import Path
from dflow.python import OP, OPIO, Artifact, OPIOSign


class DownloadCIF(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "api-key": str,
                "mp-id": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "DownloadCIF_output": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(self, op_in: OPIO):
        from pymatgen.ext.matproj import MPRester
        from pymatgen.io.cif import CifWriter

        api_key = op_in["api-key"]
        mp_id = str(op_in["mp-id"])
        mpr = MPRester(str(api_key))
        structure = mpr.get_structure_by_material_id(
            f"mp-{mp_id}", final=True, conventional_unit_cell=True
        )
        CifWriter(structure).write_file(f"{mp_id}.cif")

        return OPIO({"DownloadCIF_output": Path(f"{mp_id}.cif")})
