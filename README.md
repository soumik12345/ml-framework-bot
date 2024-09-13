# ML Framework Bot

An agentic bot to translate machine learning codebases across ML frameworks.

## Build Vector Index for [keras.io](https://keras.io/)

```python
from dotenv import load_dotenv

import wandb
from ml_frameworks_bot.keras_io import KerasIORetreiver

load_dotenv()
wandb.init(
    project="ml-frameworks-bot", entity="geekyrakshit", job_type="build_vector_index"
)
retriever = KerasIORetreiver(
    embedding_model_name="BAAI/bge-small-en-v1.5", repository_local_path="./keras_docs"
)
vector_index = retriever.index_documents(
    vector_index_persist_dir="vector_indices/keras_docs_vector_index_from_documents",
    artifact_name="keras_docs_vector_index",
)
```

## Load [keras.io](https://keras.io/) Retreiver from Vector Index

```python
import weave
from dotenv import load_dotenv

from ml_frameworks_bot.keras_io import KerasIORetreiver

load_dotenv()
weave.init(project_name="geekyrakshit/ml-frameworks-bot")
retriever = KerasIORetreiver.from_wandb_artifact(
    artifact_address="geekyrakshit/ml-frameworks-bot/keras_docs_vector_index:latest"
)
nodes = retriever.predict(
    query="""
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
"""
)
```