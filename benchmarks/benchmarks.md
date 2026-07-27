Primitive benchmarks
matrix-vector
matrix-matrix
elementwise activation
normalization
transpose or layout conversion

Test several shape classes, not only one network.

Operator benchmarks
dense
dense + activation
QKV projection
attention
feedforward block
End-to-end benchmarks
small MLP
transformer block
several sequence lengths
batch 1 versus larger batches

Measure at least:

latency
throughput
bytes allocated or reserved
peak workspace
effective memory bandwidth
generated binary size

---

The new benchmark harness cannot currently be trusted.

     Several controls are disconnected:
      - benchmarks/benchmark.zig:131 calculates the selected benchmark,
        but main ignores it and always registers all three benchmarks.

      - Even -Dbenchmark=batch_sweep ran inference and both matrix
        benchmarks.

      - benchmarks/benchmark.zig:214 hardcodes false, so “batched” and
        “no batch” execute the same implementation.

      - In my run they accordingly measured almost identically:
        approximately 2.02 ms per 1,000 multiplies.

      - -Dwrite_out is unused, so the prior CSV workflow no longer
        works.

      - Historical CSV results use different measurement semantics than
        zbench, making before/after comparisons unsafe without a
        migration baseline.