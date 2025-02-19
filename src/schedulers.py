from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, LCMScheduler

from src.registry import schedulers_registry


@schedulers_registry.add_to_registry("dpm_solver_scheduler")
class DPMSolverScheduler(DPMSolverMultistepScheduler):
    pass


@schedulers_registry.add_to_registry("ddim_scheduler")
class DDIMScheduler(DDIMScheduler):
    pass


@schedulers_registry.add_to_registry("lcm_scheduler")
class LCMScheduler(LCMScheduler):
    pass
