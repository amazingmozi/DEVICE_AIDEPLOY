"""MLP model definition for the sine regression task."""

import tensorflow as tf


def build_mlp(input_shape=(1,), hidden_units=(16, 16), learning_rate=1e-2):
    """Build and compile a simple MLP regression model."""
    model = tf.keras.Sequential(name="sine_mlp")
    model.add(tf.keras.layers.Input(shape=input_shape, name="x"))

    for index, units in enumerate(hidden_units, start=1):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{index}"))

    model.add(tf.keras.layers.Dense(1, name="sin_prediction"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
