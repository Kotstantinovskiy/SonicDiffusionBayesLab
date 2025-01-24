from diffusers import StableDiffusionPipeline


class StableDiffusionModel(StableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
