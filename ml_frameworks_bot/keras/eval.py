import weave
from weave import Dataset

# Initialize Weave
weave.init(project_name="ml-colabs/ml-frameworks-bot")

# Create a dataset
eval_dataset = Dataset(
    name="keras_evaluation_dataset",
    rows=[
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                model = keras.Sequential(\n                    [\n                        keras.Input(shape=input_shape),\n                        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),\n                        layers.MaxPooling2D(pool_size=(2, 2)),\n                        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),\n                        layers.MaxPooling2D(pool_size=(2, 2)),\n                        layers.Flatten(),\n                        layers.Dropout(0.5),\n                        layers.Dense(num_classes, activation="softmax"),\n                    ]\n                )\n\n                model.summary()\n            ',
            "keras_ops": [
                "keras.layers.Dense",
                "keras.Input",
                "keras.layers.MaxPooling2D",
                "keras.layers.Flatten",
                "keras.layers.Conv2D",
                "keras.Sequential",
                "keras.Model.summary",
                "keras.layers.Dropout",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/core_layers/dense.md",
                "sources/api/layers/core_layers/input.md",
                "sources/api/layers/pooling_layers/max_pooling2d.md",
                "sources/api/layers/reshaping_layers/flatten.md",
                "sources/api/layers/convolution_layers/convolution2d.md",
                "sources/api/models/sequential.md",
                "sources/api/models/model.md",
                "sources/api/layers/regularization_layers/dropout.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                model = keras.Sequential(\n                [\n                    layers.Dense(2, activation="relu"),\n                    layers.Dense(3, activation="relu"),\n                    layers.Dense(4),\n                ]\n            )\n            ',
            "keras_ops": ["keras.layers.Dense", "keras.Sequential"],
            "keras_api_reference_path": [
                "sources/api/layers/core_layers/dense.md",
                "sources/api/models/sequential.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                data_augmentation = keras.Sequential(\n                    [\n                        layers.Normalization(),\n                        layers.Resizing(image_size, image_size),\n                        layers.RandomFlip("horizontal"),\n                        layers.RandomRotation(factor=0.02),\n                        layers.RandomZoom(height_factor=0.2, width_factor=0.2),\n                    ],\n                    name="data_augmentation",\n                )\n            ',
            "keras_ops": [
                "keras.layers.Normalization",
                "keras.layers.Resizing",
                "keras.layers.RandomFlip",
                "keras.layers.RandomRotation",
                "keras.layers.RandomZoom",
                "keras.Sequential",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/preprocessing_layers/numerical/normalization.md",
                "sources/api/layers/preprocessing_layers/image_preprocessing/resizing.md",
                "sources/api/layers/preprocessing_layers/image_augmentation/random_flip.md",
                "sources/api/layers/preprocessing_layers/image_augmentation/random_rotation.md",
                "sources/api/layers/preprocessing_layers/image_augmentation/random_zoom.md",
                "sources/api/models/sequential.md",
            ],
        },
        {
            "code_snippet": "\n                from keras import layers\n\n                class Patches(layers.Layer):\n                    def __init__(self, patch_size, **kwargs):\n                        super().__init__(**kwargs)\n                        self.patch_size = patch_size\n\n                    def call(self, x):\n                        patches = keras.ops.image.extract_patches(x, self.patch_size)\n                        batch_size = keras.ops.shape(patches)[0]\n                        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]\n                        patch_dim = keras.ops.shape(patches)[3]\n                        out = keras.ops.reshape(patches, (batch_size, num_patches, patch_dim))\n                        return out\n            ",
            "keras_ops": [
                "keras.layers.Layer",
                "keras.ops.image.extract_patches",
                "keras.ops.shape",
                "keras.ops.reshape",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/base_layer.md",
                "sources/api/ops/image/index.md",
                "sources/api/ops/core/index.md",
                "sources/api/ops/numpy/index.md",
            ],
        },
        {
            "code_snippet": '\n                from keras import layers\n\n                def conv_block(filters, inputs):\n                    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)\n                    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)\n                    x = layers.BatchNormalization()(x)\n                    outputs = layers.MaxPool2D()(x)\n\n                    return outputs\n            ',
            "keras_ops": [
                "keras.layers.SeparableConv2D",
                "keras.layers.BatchNormalization",
                "keras.layers.MaxPool2D",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/convolution_layers/separable_convolution2d.md",
                "sources/api/layers/normalization_layers/batch_normalization.md",
                "sources/api/layers/pooling_layers/max_pooling2d.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n\n                METRICS = [\n                    keras.metrics.BinaryAccuracy(),\n                    keras.metrics.Precision(name="precision"),\n                    keras.metrics.Recall(name="recall"),\n                ]\n            ',
            "keras_ops": [
                "keras.metrics.BinaryAccuracy",
                "keras.metrics.Precision",
                "keras.metrics.Recall",
            ],
            "keras_api_reference_path": [
                "sources/api/metrics/accuracy_metrics.md",
                "sources/api/metrics/classification_metrics.md",
                "sources/api/metrics/classification_metrics.md",
            ],
        },
        {
            "code_snippet": "\n                import keras\n                from keras import layers\n\n                class RandomColorAffine(layers.Layer):\n                    def __init__(self, brightness=0, jitter=0, **kwargs):\n                        super().__init__(**kwargs)\n\n                        self.seed_generator = keras.random.SeedGenerator(1337)\n                        self.brightness = brightness\n                        self.jitter = jitter\n\n                    def call(self, images, training=True):\n                        if training:\n                            batch_size = ops.shape(images)[0]\n\n                            # Same for all colors\n                            brightness_scales = 1 + keras.random.uniform(\n                                (batch_size, 1, 1, 1),\n                                minval=-self.brightness,\n                                maxval=self.brightness,\n                                seed=self.seed_generator,\n                            )\n                            # Different for all colors\n                            jitter_matrices = keras.random.uniform(\n                                (batch_size, 1, 3, 3), \n                                minval=-self.jitter, \n                                maxval=self.jitter,\n                                seed=self.seed_generator,\n                            )\n\n                            color_transforms = (\n                                ops.tile(ops.expand_dims(ops.eye(3), axis=0), (batch_size, 1, 1, 1))\n                                * brightness_scales\n                                + jitter_matrices\n                            )\n                            images = ops.clip(ops.matmul(images, color_transforms), 0, 1)\n                        return images\n            ",
            "keras_ops": [
                "keras.layers.Layer",
                "keras.random.SeedGenerator",
                "keras.random.uniform",
                "keras.ops.shape",
                "keras.ops.tile",
                "keras.ops.expand_dims",
                "keras.ops.eye",
                "keras.ops.clip",
                "keras.ops.matmul",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/base_layer.md",
                "sources/api/random/seed_generator.md",
                "sources/api/random/random_ops.md",
                "sources/api/ops/core/index.md",
                "sources/api/ops/numpy/index.md",
                "sources/api/ops/numpy/index.md",
                "sources/api/ops/numpy/index.md",
                "sources/api/ops/numpy/index.md",
                "sources/api/ops/numpy/index.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                def build_dce_net():\n                    input_img = keras.Input(shape=[None, None, 3])\n                    conv1 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(input_img)\n                    conv2 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(conv1)\n                    conv3 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(conv2)\n                    conv4 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(conv3)\n                    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])\n                    conv5 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(int_con1)\n                    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])\n                    conv6 = layers.Conv2D(\n                        32, (3, 3), strides=(1, 1), activation="relu", padding="same"\n                    )(int_con2)\n                    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])\n                    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(\n                        int_con3\n                    )\n                    return keras.Model(inputs=input_img, outputs=x_r)\n            ',
            "keras_ops": [
                "keras.layers.Conv2D",
                "keras.layers.Concatenate",
                "keras.Model",
                "keras.Input",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/convolution_layers/convolution2d.md",
                "sources/api/layers/merging_layers/concatenate.md",
                "sources/api/models/model.md",
                "sources/api/layers/core_layers/input.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                inputs = keras.Input(shape=(None,), dtype="int32")\n                x = layers.Embedding(max_features, 128)(inputs)\n                x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)\n                x = layers.Bidirectional(layers.LSTM(64))(x)\n                outputs = layers.Dense(1, activation="sigmoid")(x)\n                model = keras.Model(inputs, outputs)\n            ',
            "keras_ops": [
                "keras.layers.Embedding",
                "keras.layers.Bidirectional",
                "keras.layers.LSTM",
                "keras.layers.Dense",
                "keras.Model",
                "keras.Input",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/core_layers/embedding.md",
                "sources/api/layers/recurrent_layers/bidirectional.md",
                "sources/api/layers/recurrent_layers/lstm.md",
                "sources/api/layers/core_layers/dense.md",
                "sources/api/models/model.md",
                "sources/api/layers/core_layers/input.md",
            ],
        },
        {
            "code_snippet": '\n                import keras\n                from keras import layers\n\n                class TransformerEncoder(layers.Layer):\n                    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):\n                        super().__init__()\n                        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)\n                        self.ffn = keras.Sequential(\n                            [\n                                layers.Dense(feed_forward_dim, activation="relu"),\n                                layers.Dense(embed_dim),\n                            ]\n                        )\n                        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)\n                        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)\n                        self.dropout1 = layers.Dropout(rate)\n                        self.dropout2 = layers.Dropout(rate)\n\n                    def call(self, inputs, training=False):\n                        attn_output = self.att(inputs, inputs)\n                        attn_output = self.dropout1(attn_output, training=training)\n                        out1 = self.layernorm1(inputs + attn_output)\n                        ffn_output = self.ffn(out1)\n                        ffn_output = self.dropout2(ffn_output, training=training)\n                        return self.layernorm2(out1 + ffn_output)\n            ',
            "keras_ops": [
                "keras.layers.Layer",
                "keras.layers.MultiHeadAttention",
                "keras.Sequential",
                "keras.layers.Dense",
                "keras.layers.LayerNormalization",
                "keras.layers.Dropout",
            ],
            "keras_api_reference_path": [
                "sources/api/layers/base_layer.md",
                "sources/api/layers/attention_layers/multi_head_attention.md",
                "sources/api/models/sequential.md",
                "sources/api/layers/core_layers/dense.md",
                "sources/api/layers/normalization_layers/layer_normalization.md",
                "sources/api/layers/regularization_layers/dropout.md",
            ],
        },
    ],
)

# Publish the dataset
weave.publish(eval_dataset)
