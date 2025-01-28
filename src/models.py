import time

from diffusers import StableDiffusionPipeline

from src.registry import models_registry


@models_registry.add_to_registry("stable_diffusion_model")
class StableDiffusionModel(StableDiffusionPipeline):
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = super().__call__(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
