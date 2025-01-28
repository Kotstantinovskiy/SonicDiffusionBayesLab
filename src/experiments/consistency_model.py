from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("consistency_model")
class ConsistencyModelMethod(BaseMethod):
    def __init__(self):
        pass
