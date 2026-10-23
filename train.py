"""Train and evaluate a simple MLP on generated sine-wave data."""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_generate import create_train_test_data, save_data_preview
from model.template_sin import build_mlp


CONFIG = {
    "train_samples": 1000,
    "test_samples": 200,
    "noise_std": 0.1,
    "seed": 42,
    "hidden_units": (16, 16),
    "learning_rate": 1e-2,
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "tolerance": 0.15,
}

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "train.log"
MODEL_PATH = Path("checkpoints/sine_model.h5")
PLOT_PATH = Path("training_data.png")


def setup_logger():
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("device_AIDeploy.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def tolerance_accuracy(y_true, y_pred, tolerance):
    """Regression accuracy: ratio of predictions within an absolute error tolerance."""
    errors = np.abs(y_true.reshape(-1) - y_pred.reshape(-1))
    return float(np.mean(errors <= tolerance))


def main():
    logger = setup_logger()
    tf.keras.utils.set_random_seed(CONFIG["seed"])

    logger.info("Training configuration: %s", CONFIG)
    x_train, y_train, x_test, y_test = create_train_test_data(
        train_samples=CONFIG["train_samples"],
        test_samples=CONFIG["test_samples"],
        noise_std=CONFIG["noise_std"],
        seed=CONFIG["seed"],
    )
    save_data_preview(x_train, y_train, PLOT_PATH)

    logger.info("Training samples: %s", len(x_train))
    logger.info("Test samples: %s", len(x_test))
    logger.info("Input shape: %s", x_train.shape)
    logger.info("Output shape: %s", y_train.shape)

    model = build_mlp(
        input_shape=x_train.shape[1:],
        hidden_units=CONFIG["hidden_units"],
        learning_rate=CONFIG["learning_rate"],
    )

    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.CSVLogger(str(LOG_DIR / "history.csv")),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["validation_split"],
        callbacks=callbacks,
        verbose=0,
    )

    for epoch, (loss, mae, val_loss, val_mae) in enumerate(
        zip(
            history.history["loss"],
            history.history["mae"],
            history.history["val_loss"],
            history.history["val_mae"],
        ),
        start=1,
    ):
        logger.info(
            "Epoch %03d/%03d - loss: %.6f - mae: %.6f - val_loss: %.6f - val_mae: %.6f",
            epoch,
            CONFIG["epochs"],
            loss,
            mae,
            val_loss,
            val_mae,
        )

    test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    test_accuracy = tolerance_accuracy(y_test, predictions, CONFIG["tolerance"])

    logger.info("Evaluation:")
    logger.info("Test loss (MSE): %.6f", test_loss)
    logger.info("Test MAE: %.6f", test_mae)
    logger.info("Test RMSE: %.6f", test_rmse)
    logger.info("Tolerance accuracy (abs error <= %.2f): %.4f", CONFIG["tolerance"], test_accuracy)

    model.save(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved training log to %s", LOG_FILE)
    logger.info("Saved epoch history to %s", LOG_DIR / "history.csv")
    logger.info("Saved data preview to %s", PLOT_PATH)


if __name__ == "__main__":
    main()
