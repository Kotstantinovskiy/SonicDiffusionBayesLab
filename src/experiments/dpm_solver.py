from base_experiment import BaseMethod

from src.registry import methods_registry


@methods_registry.add_to_registry("dpm_solver")
class DPMSolverMethod(BaseMethod):
    def __init__(self):
        pass
