# ML Framework Bot

An agentic workflow to translate machine learning codebases across ML frameworks reliably at scale.

## Installation

```bash
uv sync
```

<details>
<summary>Build Vector Index for <a href="https://keras.io/">keras.io</a></summary>
  
```python
import torch
from dotenv import load_dotenv

import wandb
from ml_frameworks_bot.keras_io import KerasIORetreiver

load_dotenv()
wandb.init(
    project="ml-frameworks-bot", entity="ml-colabs", job_type="build_vector_index"
)
retriever = KerasIORetreiver(
    embedding_model_name="BAAI/bge-small-en-v1.5",
    torch_dtype=torch.float16,
    repository_local_path="keras_docs",
)
vector_index = retriever.index_documents(
    included_directories=["sources/api"],
    vector_index_persist_dir="vector_indices/keras3_api_reference",
    artifact_name="keras3_api_reference",
)
```
</details>

<details>
<summary>Load <a href="https://keras.io/">keras.io</a> Retreiver from Vector Index and perform Retrieval</summary>
  
```python
import weave
from dotenv import load_dotenv

from ml_frameworks_bot.keras_io import KerasIORetreiver

load_dotenv()
weave.init(project_name="ml-colabs/ml-frameworks-bot")
retriever = KerasIORetreiver.from_wandb_artifact(
    artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
)
retrieved_nodes = retriever.predict(
    query="Fetch the API referece for `keras.layers.Dense`"
)
```
</details>

<details>
<summary>Run a Keras Documentation Agent</summary>
  
```python
import weave
from dotenv import load_dotenv

from ml_frameworks_bot.keras_io import KerasDocumentationAgent, KerasIORetreiver

load_dotenv()
weave.init(project_name="geekyrakshit/ml-frameworks-bot")
template_retriever = KerasIORetreiver.from_wandb_artifact(
    artifact_address="geekyrakshit/ml-frameworks-bot/keras_io_vector_index_templates:latest"
)
guides_retriever = KerasIORetreiver.from_wandb_artifact(
    artifact_address="geekyrakshit/ml-frameworks-bot/keras_io_vector_index_templates:latest"
)
example_retriever = KerasIORetreiver.from_wandb_artifact(
    artifact_address="geekyrakshit/ml-frameworks-bot/keras_io_vector_index_examples:latest"
)
agent = KerasDocumentationAgent(
    llm_name="gpt-4o",
    template_retriever=template_retriever,
    guides_retriever=guides_retriever,
    example_retriever=example_retriever,
)
response = agent.predict(
    code_snippet="""
import keras
from keras import layers

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
""",
    seed=42,
)
```
</details>