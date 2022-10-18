package ffx.numerics.benchmark;

import static java.lang.Math.random;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static jdk.incubator.vector.DoubleVector.fromArray;
import static jdk.incubator.vector.DoubleVector.zero;
import static jdk.incubator.vector.VectorOperators.ADD;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Warmup;

public class PanamaBenchmark {

  /**
   * Perform 5 test warm-up iterations
   */
  private final static int warmUpIterations = 5;
  /**
   * Each warm-up iteration will run for this many seconds.
   */
  private final static int warmupTime = 1;
  /**
   * Perform 5 test measurement iterations
   */
  private final static int measurementIterations = 5;
  /**
   * Each measurement iteration will run for this many seconds.
   */
  private final static int measurementTime = 1;

  private static final VectorSpecies SPECIES = DoubleVector.SPECIES_PREFERRED;

  private static final int size = 2048;
  private static final double[] left = new double[size];
  private static final double[] right = new double[size];

  static {
    for (int i=0; i<size; i++) {
      left[i] = random();
      right[i] = random();
    }
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vectorfma() {
    var sum = DoubleVector.zero(SPECIES);
    for (int i = 0; i < size; i += SPECIES.length()) {
      var l = fromArray(SPECIES, left, i);
      var r = fromArray(SPECIES, right, i);
      sum = l.fma(r, sum);
    }
    return sum.reduceLanes(ADD);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vectorfmaUnrolled() {
    var sum1 = zero(SPECIES);
    var sum2 = zero(SPECIES);
    var sum3 = zero(SPECIES);
    var sum4 = zero(SPECIES);
    int width = SPECIES.length();
    for (int i = 0; i < size; i += width * 4) {
      sum1 = fromArray(SPECIES, left, i).fma(fromArray(SPECIES, right, i), sum1);
      sum2 = fromArray(SPECIES, left, i + width).fma(fromArray(SPECIES, right, i + width), sum2);
      sum3 = fromArray(SPECIES, left, i + width * 2).fma(fromArray(SPECIES, right, i + width * 2), sum3);
      sum4 = fromArray(SPECIES, left, i + width * 3).fma(fromArray(SPECIES, right, i + width * 3), sum4);
    }
    return sum1.add(sum2).add(sum3).add(sum4).reduceLanes(ADD);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vector() {
    var sum = zero(SPECIES);
    for (int i = 0; i < size; i += SPECIES.length()) {
      var l = fromArray(SPECIES, left, i);
      var r = fromArray(SPECIES, right, i);
      sum = l.mul(r).add(sum);
    }
    return sum.reduceLanes(ADD);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vectorUnrolled() {
    var sum1 = zero(SPECIES);
    var sum2 = zero(SPECIES);
    var sum3 = zero(SPECIES);
    var sum4 = zero(SPECIES);
    int width = SPECIES.length();
    for (int i = 0; i < size; i += width * 4) {
      sum1 = fromArray(SPECIES, left, i).mul(fromArray(SPECIES, right, i)).add(sum1);
      sum2 = fromArray(SPECIES, left, i + width).mul(fromArray(SPECIES, right, i + width)).add(sum2);
      sum3 = fromArray(SPECIES, left, i + width * 2).mul(fromArray(SPECIES, right, i + width * 2)).add(sum3);
      sum4 = fromArray(SPECIES, left, i + width * 3).mul(fromArray(SPECIES, right, i + width * 3)).add(sum4);
    }
    return sum1.add(sum2).add(sum3).add(sum4).reduceLanes(ADD);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double unrolled() {
    double s0 = 0f;
    double s1 = 0f;
    double s2 = 0f;
    double s3 = 0f;
    // double s4 = 0f;
    // double s5 = 0f;
    // double s6 = 0f;
    // double s7 = 0f;
    for (int i = 0; i < size; i += 8) {
      s0 = Math.fma(left[i + 0],  right[i + 0], s0);
      s1 = Math.fma(left[i + 1],  right[i + 1], s1);
      s2 = Math.fma(left[i + 2],  right[i + 2], s2);
      s3 = Math.fma(left[i + 3],  right[i + 3], s3);
      // s4 = Math.fma(left[i + 4],  right[i + 4], s4);
      // s5 = Math.fma(left[i + 5],  right[i + 5], s5);
      // s6 = Math.fma(left[i + 6],  right[i + 6], s6);
      // s7 = Math.fma(left[i + 7],  right[i + 7], s7);
    }
    // return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    return s0 + s1 + s2 + s3;
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vanilla() {
    double sum = 0f;
    for (int i = 0; i < size; ++i) {
      sum = Math.fma(left[i], right[i], sum);
    }
    return sum;
  }

}
