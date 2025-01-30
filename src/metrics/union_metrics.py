"""'
from src.metrics.metrics import ClipScoreMetric, RewardModel, FID  # Import your metrics here
from src.registry import metrics_registry


class UnionMetrics:
    def __init__(self, config, metrics: list = None):
        if not metrics:
            self.metrics = {metric: metrics_registry[metric] for metric in metrics_registry.keys()}

        self.metrics = {metric: metrics_registry[metric] for metric in metrics}

    def update_scores(self, pred_images, real_images, prompts):
        if 'clip_score' in self.metrics:
            self.metrics['clip_score'].update(pred_images, prompts)

        scores = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            scores[metric_name] = metric.compute(data)
        return scores
"""
# Example usage:
# union_metrics = UnionMetrics()
# data = ...  # Your data here
# scores = union_metrics.compute_scores(data)
# print(scores)
