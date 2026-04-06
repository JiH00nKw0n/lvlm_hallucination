from src.common import EvaluateConfig, TrainConfig, setup_logger
from src.utils import now, load_yml


def setup_task(*args, **kwargs):
    # Lazy import to avoid pulling optional training dependencies at package import time.
    from src.tasks import setup_task as _setup_task
    return _setup_task(*args, **kwargs)

__all__ = [
    "EvaluateConfig",
    "TrainConfig",
    "setup_logger",
    "now",
    "load_yml",
    "setup_task",
]
