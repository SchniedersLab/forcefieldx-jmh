package ffx.numerics.benchmark;

import static java.lang.Math.fma;
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
  private final static int warmUpIterations = 2;
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

  private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

  private static final int size = 2048;
  private static final double[] left = new double[size];
  private static final double[] right = new double[size];

  static {
    for (int i = 0; i < size; i++) {
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
  public double vanilla() {
    double sum = 0f;
    for (int i = 0; i < size; ++i) {
      sum += left[i] * right[i];
    }
    return sum;
  }

  /**
   * 4 vs. 8 in the loop give similar timings.
   */
  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double unrolled8() {
    double s0 = 0f;
    double s1 = 0f;
    double s2 = 0f;
    double s3 = 0f;
    double s4 = 0f;
    double s5 = 0f;
    double s6 = 0f;
    double s7 = 0f;
    for (int i = 0; i < size; i += 8) {
      s0 += left[i + 0] * right[i + 0];
      s1 += left[i + 1] * right[i + 1];
      s2 += left[i + 2] * right[i + 2];
      s3 += left[i + 3] * right[i + 3];
      s4 += left[i + 4] * right[i + 4];
      s5 += left[i + 5] * right[i + 5];
      s6 += left[i + 6] * right[i + 6];
      s7 += left[i + 7] * right[i + 7];
    }
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double unrolledfma8() {
    return dot(left, right);
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
  public double vectorWithMask() {
    var sum = zero(SPECIES);
    for (int i = 0; i < size; i += SPECIES.length()) {
      var mask = SPECIES.indexInRange(i, size);
      var l = fromArray(SPECIES, left, i, mask);
      var r = fromArray(SPECIES, right, i, mask);
      sum = l.mul(r, mask).add(sum, mask);
    }
    return sum.reduceLanes(ADD);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public double vectorUnrolled4() {
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
  public double vectorfmaUnrolled4() {
    return vectorDot(left, right);
  }



  /**
   * Compute the dot product of vectors a and b.
   *
   * @param a The first vector.
   * @param b The second vector.
   * @return The dot product.
   */
  public static double dot(double[] a, double[] b) {
    return dot(a, b, 0, a.length);
  }

  /**
   * Compute the dot product of vectors a and b.
   *
   * @param a The first vector.
   * @param b The second vector.
   * @param start The first index to evaluate.
   * @param length The number of positions to include.
   * @return The dot product.
   */
  public static double dot(double[] a, double[] b, int start, int length) {
    var s0 = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    var s3 = 0.0;
    var s4 = 0.0;
    var s5 = 0.0;
    var s6 = 0.0;
    var s7 = 0.0;
    int stop = start + length;
    int step = 8;
    int i = start;
    for (; i <= stop - step; i += step) {
      s0 = fma(a[i + 0], b[i + 0], s0);
      s1 = fma(a[i + 1], b[i + 1], s1);
      s2 = fma(a[i + 2], b[i + 2], s2);
      s3 = fma(a[i + 3], b[i + 3], s3);
      s4 = fma(a[i + 4], b[i + 4], s4);
      s5 = fma(a[i + 5], b[i + 5], s5);
      s6 = fma(a[i + 6], b[i + 6], s6);
      s7 = fma(a[i + 7], b[i + 7], s7);
    }
    double vectorSum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

    // Finish up.
    for (; i < stop; i++) {
      vectorSum = fma(a[i], b[i], vectorSum);
    }

    return vectorSum;
  }

  /**
   * Uses jdk.incubator.vector API to vectorize the dot product.
   *
   * @param a The first vector.
   * @param b The second vector.
   * @return The dot product.
   */
  public static double vectorDot(double[] a, double[] b) {
    return vectorDot(a, b, 0, a.length);
  }

  /**
   * Uses jdk.incubator.vector API to vectorize the dot product.
   *
   * @param a The first vector.
   * @param b The second vector.
   * @param start The first index to evaluate.
   * @param length The number of positions to include.
   * @return The dot product.
   */
  public static double vectorDot(double[] a, double[] b, int start, int length) {
    var sum1 = zero(SPECIES);
    var sum2 = zero(SPECIES);
    var sum3 = zero(SPECIES);
    var sum4 = zero(SPECIES);
    int width = SPECIES.length();
    int size = start + length;
    int step = width * 4;
    int i = start;
    for (; i <= size - step; i += step) {
      sum1 = fromArray(SPECIES, a, i).fma(fromArray(SPECIES, b, i), sum1);
      sum2 = fromArray(SPECIES, a, i + width).fma(fromArray(SPECIES, b, i + width), sum2);
      sum3 = fromArray(SPECIES, a, i + width * 2).fma(fromArray(SPECIES, b, i + width * 2), sum3);
      sum4 = fromArray(SPECIES, a, i + width * 3).fma(fromArray(SPECIES, b, i + width * 3), sum4);
    }
    double vectorSum = sum1.add(sum2).add(sum3).add(sum4).reduceLanes(ADD);

    // Finish up.
    for (; i < size; i++) {
      vectorSum = fma(a[i], b[i], vectorSum);
    }

    return vectorSum;
  }

}
