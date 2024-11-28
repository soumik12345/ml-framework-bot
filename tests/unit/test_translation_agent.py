import pytest

from ml_frameworks_bot.translation_agent import FrameworkIdentificationModel


@pytest.mark.parametrize(
    "code_snippet, source_framework",
    [
        (
            """
            import tensorflow as tf
            from tensorflow import keras
            """,
            "keras2",
        ),
        (
            "import keras",
            "keras3",
        ),
        (
            "import torch.nn as nn",
            "pytorch",
        ),
        (
            "import mlx.nn as nn",
            "mlx",
        ),
        (
            "from flax import nnx",
            "flax",
        ),
        (
            "import jax.numpy as jnp",
            "jax",
        ),
        (
            "import numpy as np",
            "numpy",
        ),
    ],
    ids=["keras2", "keras3", "pytorch", "mlx", "flax", "jax", "numpy"],
)
def test_framework_identification_model(
    code_snippet: str, source_framework: str
) -> None:
    identification_agent = FrameworkIdentificationModel(model_name="gpt-4o-mini")
    assert identification_agent.predict(code_snippet=code_snippet) == source_framework
