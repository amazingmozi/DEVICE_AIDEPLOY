"""Dataset generation utilities for the sine regression task."""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(num_samples=1000, noise_std=0.1, seed=None):
    """Generate noisy sine-wave samples."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, num_samples).astype(np.float32)
    y = np.sin(x) + rng.normal(0, noise_std, num_samples).astype(np.float32)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def create_train_test_data(train_samples=1000, test_samples=200, noise_std=0.1, seed=42):
    """Create deterministic train/test data for experiments."""
    x_train, y_train = generate_data(train_samples, noise_std=noise_std, seed=seed)
    x_test, y_test = generate_data(test_samples, noise_std=noise_std, seed=seed + 1)
    return x_train, y_train, x_test, y_test


def save_data_preview(x_train, y_train, output_path="training_data.png"):
    """Save a small scatter plot of the generated training data."""
    plt.figure(figsize=(10, 4))
    plt.scatter(x_train[:100].reshape(-1), y_train[:100].reshape(-1), alpha=0.5)
    plt.title("Training Data (first 100 samples)")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = create_train_test_data()
    save_data_preview(x_train, y_train)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape}")
    print(f"Output shape: {y_train.shape}")
