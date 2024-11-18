from typing import Any, Dict


RepositoryMapping: Dict[str, Dict[str, str]] = {
    "keras3": {
        "repository": "https://github.com/keras-team/keras-io",
        "artifact_address": "ml-colabs/ml-frameworks-bot/keras3-docs:latest",
        "mapping": "mappings/keras3.json",
    },
    "keras2": {
        "repository": "https://github.com/keras-team/keras-io",
        "artifact_address": "ml-colabs/ml-frameworks-bot/keras2-docs:latest",
    },
    "mlx": {
        "repository": "https://github.com/ml-explore/mlx",
        "artifact_address": "ml-colabs/ml-frameworks-bot/mlx-docs:latest",
        "mapping": "mappings/mlx.json",
    },
    "jax": {
        "repository": "https://github.com/jax-ml/jax",
        "artifact_address": "ml-colabs/ml-frameworks-bot/jax-docs:latest",
    },
    "flax": {
        "repository": "https://github.com/google/flax",
        "artifact_address": "ml-colabs/ml-frameworks-bot/flax-nnx-docs:latest",
    },
    "numpy": {
        "repository": "https://github.com/numpy/numpy",
        "artifact_address": "ml-colabs/ml-frameworks-bot/numpy-docs:latest",
    },
}

FrameworkParams: Dict[str, Dict[str, Any]] = {
    "keras3": {
        "chunk_on_separator": True,
        "split_pattern": [("ops", "----")],
        "included_directories": ["sources/api"],
        "exclude_file_postfixes": ["index.md"],
        "included_file_extensions": [".md"],
    },
    "mlx": {
        "chunk_on_separator": False,
        "included_directories": ["dev", "examples", "python", "usage"],
        "included_file_extensions": [".txt"],
    },
}
