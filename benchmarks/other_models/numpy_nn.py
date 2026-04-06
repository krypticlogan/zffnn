import csv
import time
from pathlib import Path

import numpy as np


# Activation functions
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Handle numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def none(x):
        return x


# Neural Network implementation from scratch
class NeuralNetwork:
    def __init__(self, layers_config, seed=500):
        """
        layers_config: list of (size, activation) tuples
        First element is input size
        Subsequent elements are layer outputs with their activations
        """
        self.layers_config = layers_config
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases
        np.random.seed(seed)

        for i in range(len(layers_config) - 1):
            in_features = layers_config[i][0]
            out_features = layers_config[i + 1][0]
            activation_name = layers_config[i + 1][1]

            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (in_features + out_features))
            w = np.random.uniform(-limit, limit, (in_features, out_features)).astype(
                np.float64
            )
            b = np.zeros(out_features, dtype=np.float64)

            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(activation_name)

    def forward(self, x):
        """
        Forward pass through the network
        x: input array of shape (batch_size, input_features)
        """
        x = x.astype(np.float64)

        for i, (w, b, activation) in enumerate(
            zip(self.weights, self.biases, self.activations)
        ):
            # Linear transformation: y = x @ W + b
            x = np.dot(x, w) + b

            # Apply activation
            if activation == "relu":
                x = Activation.relu(x)
            elif activation == "softmax":
                x = Activation.softmax(x)
            elif activation == "none":
                x = Activation.none(x)

        return x

    def count_parameters(self):
        """Count total number of parameters (weights + biases)"""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


# Model configurations matching Zig benchmark
small_model = [
    (32, "none"),
    (16, "relu"),
    (8, "softmax"),
]

medium_model = [
    (128, "none"),
    (64, "relu"),
    (10, "softmax"),
]

large_model = [
    (1024, "none"),
    (256, "relu"),
    (128, "relu"),
    (64, "relu"),
    (2, "softmax"),
]

models = [
    ("small_model", small_model),
    ("medium_model", medium_model),
    ("large_model", large_model),
]


def param_count(model_config):
    """Calculate total parameter count"""
    count = 0
    for i in range(len(model_config) - 1):
        in_features = model_config[i][0]
        out_features = model_config[i + 1][0]
        # weights + biases
        count += in_features * out_features + out_features
    return count


def model_to_string(model_config):
    """Convert model config to string representation"""
    parts = [f"{model_config[0][1]}({model_config[0][0]})"]
    for i in range(1, len(model_config)):
        activation, size = model_config[i][1], model_config[i][0]
        parts.append(f"{activation}({size})")
    return "-".join(parts)


def inference_test(model, batch_size, feature_count, iterations, seed):
    """Run inference test and return total time in nanoseconds"""
    np.random.seed(seed)

    # Create random input
    x = np.random.randn(batch_size, feature_count).astype(np.float64)

    # Warm up
    for _ in range(max(1, iterations // 10)):
        _ = model.forward(x)

    # Actual benchmark
    start = time.perf_counter_ns()

    for _ in range(iterations):
        _ = model.forward(x)

    end = time.perf_counter_ns()

    return end - start


def benchmark_inference():
    """Run inference benchmark matching Zig benchmark_inference"""
    output_file = Path("numpy_inference_benchmark.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    seed = 500
    runs_per_model = 3

    flat_model_iterations = [1_000_000, 200_000, 50_000]
    batched_model_iterations = [500_000, 200_000, 10_000]
    model_iterations = (
        flat_model_iterations if batch_size == 1 else batched_model_iterations
    )

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        for (model_name, model_config), iterations in zip(models, model_iterations):
            model = NeuralNetwork(model_config, seed=seed)
            feature_count = model_config[0][0]

            total_time_elapsed = 0.0

            max_batch_latency_ns = 0.0
            min_batch_latency_ns = float("inf")
            total_batch_latency_ns = 0.0

            max_latency_ns = 0.0
            min_latency_ns = float("inf")
            total_latency_ns = 0.0

            max_throughput = 0.0
            min_throughput = float("inf")
            total_throughput = 0.0

            for _ in range(runs_per_model):
                total_ns = inference_test(
                    model, batch_size, feature_count, iterations, seed
                )
                total_time_elapsed += total_ns / 1e9

                batch_latency_ns = total_ns / iterations
                max_batch_latency_ns = max(max_batch_latency_ns, batch_latency_ns)
                min_batch_latency_ns = min(min_batch_latency_ns, batch_latency_ns)
                total_batch_latency_ns += batch_latency_ns

                total_inferences = iterations * batch_size
                latency_ns = total_ns / total_inferences
                max_latency_ns = max(max_latency_ns, latency_ns)
                min_latency_ns = min(min_latency_ns, latency_ns)
                total_latency_ns += latency_ns

                inferences_per_sec = 1e9 / latency_ns
                max_throughput = max(max_throughput, inferences_per_sec)
                min_throughput = min(min_throughput, inferences_per_sec)
                total_throughput += inferences_per_sec

            avg_batch_latency_ns = total_batch_latency_ns / runs_per_model
            avg_latency_ns = total_latency_ns / runs_per_model
            avg_throughput = total_throughput / runs_per_model

            model_str = model_to_string(model_config)
            param_ct = param_count(model_config)

            print(f"\nModel: {model_str}")
            print(f"Total time elapsed: {total_time_elapsed:.2f} sec")
            print(f"Runs per model: {runs_per_model}")
            print(f"Iterations per run: {iterations}")
            print(f"Batch size: {batch_size}")
            print("\nLatency (ns/inference):")
            print(f"\tmin: {min_latency_ns:.2f}")
            print(f"\tavg: {avg_latency_ns:.2f}")
            print(f"\tmax: {max_latency_ns:.2f}")
            print("\nBatch latency (ns/batch):")
            print(f"\tmin: {min_batch_latency_ns:.2f}")
            print(f"\tavg: {avg_batch_latency_ns:.2f}")
            print(f"\tmax: {max_batch_latency_ns:.2f}")
            print("\nThroughput (inferences/sec):")
            print(f"\tmin: {min_throughput:.2f}")
            print(f"\tavg: {avg_throughput:.2f}")
            print(f"\tmax: {max_throughput:.2f}")
            print("________________________________\n")

            # Write CSV
            writer.writerow(
                [
                    model_str,
                    param_ct,
                    "cpu",
                    batch_size,
                    f"{min_latency_ns:.2f}",
                    f"{avg_latency_ns:.2f}",
                    f"{max_latency_ns:.2f}",
                    f"{min_throughput:.2f}",
                    f"{avg_throughput:.2f}",
                    f"{max_throughput:.2f}",
                ]
            )


def batch_sweep():
    """Run batch sweep benchmark matching Zig batch_sweep"""
    output_file = Path("numpy_batchsweep_benchmark.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use medium model for batch sweep
    model_name, model_config = models[2]  # medium_model
    feature_count = model_config[0][0]

    batch_sizes = [1, 4, 16, 32, 64, 128, 512, 1024]
    iterations_per_batch = [1_000_000, 20_000, 20_000, 20_000, 20_000, 10_000, 10_000, 10_000]
    seed = 500
    runs_per_model = 2

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        for batch_size, iterations in zip(batch_sizes, iterations_per_batch):
            model = NeuralNetwork(model_config, seed=seed)

            total_time_elapsed = 0.0

            max_batch_latency_ns = 0.0
            min_batch_latency_ns = float("inf")
            total_batch_latency_ns = 0.0

            max_latency_ns = 0.0
            min_latency_ns = float("inf")
            total_latency_ns = 0.0

            max_throughput = 0.0
            min_throughput = float("inf")
            total_throughput = 0.0

            for _ in range(runs_per_model):
                total_ns = inference_test(
                    model, batch_size, feature_count, iterations, seed
                )
                total_time_elapsed += total_ns / 1e9

                batch_latency_ns = total_ns / iterations
                max_batch_latency_ns = max(max_batch_latency_ns, batch_latency_ns)
                min_batch_latency_ns = min(min_batch_latency_ns, batch_latency_ns)
                total_batch_latency_ns += batch_latency_ns

                total_inferences = iterations * batch_size
                latency_ns = total_ns / total_inferences
                max_latency_ns = max(max_latency_ns, latency_ns)
                min_latency_ns = min(min_latency_ns, latency_ns)
                total_latency_ns += latency_ns

                inferences_per_sec = 1e9 / latency_ns
                max_throughput = max(max_throughput, inferences_per_sec)
                min_throughput = min(min_throughput, inferences_per_sec)
                total_throughput += inferences_per_sec

            avg_batch_latency_ns = total_batch_latency_ns / runs_per_model
            avg_latency_ns = total_latency_ns / runs_per_model
            avg_throughput = total_throughput / runs_per_model

            model_str = model_to_string(model_config)
            param_ct = param_count(model_config)

            print(f"\nModel: {model_str}")
            print(f"Total time elapsed: {total_time_elapsed:.2f} sec")
            print(f"Runs per model: {runs_per_model}")
            print(f"Iterations per run: {iterations}")
            print(f"Batch size: {batch_size}")
            print("\nLatency (ns/inference):")
            print(f"\tmin: {min_latency_ns:.2f}")
            print(f"\tavg: {avg_latency_ns:.2f}")
            print(f"\tmax: {max_latency_ns:.2f}")
            print("\nBatch latency (ns/batch):")
            print(f"\tmin: {min_batch_latency_ns:.2f}")
            print(f"\tavg: {avg_batch_latency_ns:.2f}")
            print(f"\tmax: {max_batch_latency_ns:.2f}")
            print("\nThroughput (inferences/sec):")
            print(f"\tmin: {min_throughput:.2f}")
            print(f"\tavg: {avg_throughput:.2f}")
            print(f"\tmax: {max_throughput:.2f}")
            print("________________________________\n")

            # Write CSV
            writer.writerow(
                [
                    model_str,
                    param_ct,
                    "cpu",
                    batch_size,
                    f"{min_latency_ns:.2f}",
                    f"{avg_latency_ns:.2f}",
                    f"{max_latency_ns:.2f}",
                    f"{min_throughput:.2f}",
                    f"{avg_throughput:.2f}",
                    f"{max_throughput:.2f}",
                ]
            )


if __name__ == "__main__":
    print(f"NumPy version: {np.__version__}")

    print("\n" + "=" * 50)
    print("Running Inference Benchmark")
    print("=" * 50)
    benchmark_inference()

    print("\n" + "=" * 50)
    print("Running Batch Sweep Benchmark")
    print("=" * 50)
    batch_sweep()
