import csv
import time
from pathlib import Path

import torch
import torch.nn as nn


# Activation function mapping
class ActivationFactory:
    @staticmethod
    def create(activation_name):
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "softmax":
            return nn.Softmax(dim=1)
        elif activation_name == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")


# Model definitions matching the Zig benchmark
class NeuralNetwork(nn.Module):
    def __init__(self, layers_config):
        """
        layers_config: list of (size, activation) tuples
        First element is input size (with 'none' activation)
        Subsequent elements are layer outputs with their activations
        """
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Create linear layers between consecutive layer definitions
        for i in range(len(layers_config) - 1):
            in_features = layers_config[i][0]
            out_features = layers_config[i + 1][0]
            activation = layers_config[i + 1][1]

            self.layers.append(nn.Linear(in_features, out_features))
            self.activations.append(ActivationFactory.create(activation))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            x = activation(x)
        return x


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


def inference_test(model, device, batch_size, feature_count, iterations, seed):
    """Run inference test and return total time in nanoseconds"""
    torch.manual_seed(seed)

    # Initialize weights with seed
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    model.eval()

    with torch.no_grad():
        # Create random input
        x = torch.randn(batch_size, feature_count, device=device)

        # Warm up
        for _ in range(max(1, iterations // 10)):
            _ = model(x)

        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()

        # Actual benchmark
        start = time.perf_counter_ns()

        for _ in range(iterations):
            _ = model(x)

        # Synchronize after timing
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter_ns()

    return end - start


def benchmark_inference(device="cpu"):
    """Run inference benchmark matching Zig benchmark_inference"""
    output_file = Path("pytorch_inference_benchmark.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    seed = 500
    runs_per_model = 3

    flat_model_iterations = [20_000, 20_000, 20_000]
    batched_model_iterations = [20_000, 20_000, 10_000]
    model_iterations = (
        flat_model_iterations if batch_size == 1 else batched_model_iterations
    )

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        for (model_name, model_config), iterations in zip(models, model_iterations):
            model = NeuralNetwork(model_config).to(device)
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
                    model, device, batch_size, feature_count, iterations, seed
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
                    device,
                    batch_size,
                    f"{min_latency_ns:.2f}",
                    f"{avg_latency_ns:.2f}",
                    f"{max_latency_ns:.2f}",
                    f"{min_throughput:.2f}",
                    f"{avg_throughput:.2f}",
                    f"{max_throughput:.2f}",
                ]
            )


def batch_sweep(device="cpu"):
    """Run batch sweep benchmark matching Zig batch_sweep"""
    output_file = Path("pytorch_batchsweep_benchmark.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use medium model for batch sweep
    model_name, model_config = models[2]  # medium_model
    feature_count = model_config[0][0]

    batch_sizes = [1, 4, 16, 32, 64, 128, 512, 1024]
    iterations_per_batch = [100_000, 5_000, 5_000, 5_000, 5_000, 2500, 2500, 2500]
    seed = 500
    runs_per_model = 2

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        for batch_size, iterations in zip(batch_sizes, iterations_per_batch):
            model = NeuralNetwork(model_config).to(device)

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
                    model, device, batch_size, feature_count, iterations, seed
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
                    device,
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
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run benchmarks
    print("\n" + "=" * 50)
    print("Running Inference Benchmark")
    print("=" * 50)
    benchmark_inference(device)

    print("\n" + "=" * 50)
    print("Running Batch Sweep Benchmark")
    print("=" * 50)
    batch_sweep(device)
