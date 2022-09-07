from deepmdem.lammps import MDRelax, MDHeating, MDStabilization, MDEquilibrium
from dflow import Step, Workflow, upload_artifact
from dflow.python import PythonOPTemplate
from pathlib import Path
from dflow import SlurmRemoteExecutor
from dflow import Workflow


def md_workflow(
    temp: float,
    element_map: dict,
    running_cores: int,
    nodes: int,
    add_vacuum: float,
    potential: Path,
    input_structure: Path,
    excutor: SlurmRemoteExecutor = None,
) -> Workflow:
    """work flow for running md at target temperature"""
    potential = upload_artifact(potential)
    input_structure = upload_artifact(input_structure)

    step0 = Step(
        name="Relax",
        template=PythonOPTemplate(MDRelax, command=["source ~/dflow.env && python"]),
        parameters={
            "pbc": True,
            "element_map": element_map,
            "potential_type": "deepmd",
            "running_cores": running_cores,
        },
        artifacts={"atomic_potential": potential, "structure": input_structure},
        executor=excutor(nodes),
    )

    step1 = Step(
        name="Heating",
        template=PythonOPTemplate(MDHeating, command=["source ~/dflow.env && python"]),
        parameters={
            "temp": temp,
            "nsteps": 3000,
            "ensemble": "npt",
            "pres": 1,
            "pbc": True,
            "timestep": 0.005,
            "neidelay": 10,
            "trj_freq": 100,
            "element_map": element_map,
            "potential_type": "deepmd",
            "running_cores": running_cores,
        },
        artifacts={
            "atomic_potential": potential,
            "structure": step0.outputs.artifacts["out_structure"],
        },
        executor=excutor(nodes),
    )

    step2 = Step(
        name="Stabilization",
        template=PythonOPTemplate(
            MDStabilization, command=["source ~/dflow.env && python"]
        ),
        parameters={
            "temp": temp,
            "nsteps": 1000,
            "pbc": True,
            "timestep": 0.005,
            "neidelay": 10,
            "trj_freq": 100,
            "pres": 1,
            "element_map": element_map,
            "potential_type": "deepmd",
            "add_vacuum": add_vacuum,
            "running_cores": running_cores,
        },
        artifacts={
            "atomic_potential": potential,
            "structure": step1.outputs.artifacts["out_structure"],
        },
        executor=excutor(nodes),
    )

    step3 = Step(
        name="Equilibrium",
        template=PythonOPTemplate(
            MDEquilibrium, command=["source ~/dflow.env && python"]
        ),
        parameters={
            "temp": temp,
            "nsteps": 10000,
            "pbc": True,
            "timestep": 0.005,
            "neidelay": 10,
            "trj_freq": 10,
            "element_map": element_map,
            "potential_type": "deepmd",
            "running_cores": running_cores,
        },
        artifacts={
            "atomic_potential": potential,
            "structure": step2.outputs.artifacts["out_structure"],
        },
        executor=excutor(nodes),
    )
    wf = Workflow(name=f"md-{temp}k")
    wf.add(step0)
    wf.add(step1)
    wf.add(step2)
    wf.add(step3)
    wf.submit()

    return wf
