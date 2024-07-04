sudo java -XX:+TieredCompilation -XX:LoopUnrollLimit=0 -jar target/benchmarks.jar -prof dtraceasm:hotThreshold=0.01  MultipoleBenchmark.coulombTensorGlobalSIMD

