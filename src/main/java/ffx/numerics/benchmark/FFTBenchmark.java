
package ffx.numerics.benchmark;

import ffx.numerics.fft.Complex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Arrays;
import java.util.Random;

import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

/**
 * An example of vectorized FFT algorithm.
 */
public class FFTBenchmark {

  /**
   * Perform 2 test warm-up iterations
   */
  private final static int warmUpIterations = 2;
  /**
   * Each warm-up iteration will run for this many seconds.
   */
  private final static int warmupTime = 1;
  /**
   * Perform 3 test measurement iterations
   */
  private final static int measurementIterations = 5;
  /**
   * Each measurement iteration will run for this many seconds.
   */
  private final static int measurementTime = 1;

  public static final double[] inDouble32 = new double[32 * 2];
  public static final double[] inDouble64 = new double[64 * 2];
  public static final double[] inDouble128 = new double[128 * 2];
  public static final double[] inDouble256 = new double[256 * 2];

  // Initialize the input arrays with random values.
  static {
    Random random = new Random(1);
    for (int i = 0; i < 32; i++) {
      inDouble32[i * 2] = random.nextDouble();
    }
    for (int i = 0; i < 64; i++) {
      inDouble64[i * 2] = random.nextDouble();
    }
    for (int i = 0; i < 128; i++) {
      inDouble128[i * 2] = random.nextDouble();
    }
    for (int i = 0; i < 256; i++) {
      inDouble256[i * 2] = random.nextDouble();
    }
  }

  @State(Scope.Thread)
  public static class Complex32 {
    Complex complex = new Complex(32);
    double[] in = Arrays.copyOf(inDouble32, inDouble32.length);
  }

  @State(Scope.Thread)
  public static class Complex32Blocked {
    Complex complex = new Complex(32, 32);
    double[] in = Arrays.copyOf(inDouble32, inDouble32.length);
  }

  @State(Scope.Thread)
  public static class Complex64 {
    Complex complex = new Complex(64);
    double[] in = Arrays.copyOf(inDouble64, inDouble64.length);
  }

  @State(Scope.Thread)
  public static class Complex64Blocked {
    Complex complex = new Complex(64, 64);
    double[] in = Arrays.copyOf(inDouble64, inDouble64.length);
  }

  @State(Scope.Thread)
  public static class Complex128 {
    Complex complex = new Complex(128);
    double[] in = Arrays.copyOf(inDouble128, inDouble128.length);
  }

  @State(Scope.Thread)
  public static class Complex128Blocked {
    Complex complex = new Complex(128, 128);
    double[] in = Arrays.copyOf(inDouble128, inDouble128.length);
  }

  @State(Scope.Thread)
  public static class Complex256 {
    Complex complex = new Complex(256);
    double[] in = Arrays.copyOf(inDouble256, inDouble256.length);
  }

  @State(Scope.Thread)
  public static class Complex256Blocked {
    Complex complex = new Complex(256, 256);
    double[] in = Arrays.copyOf(inDouble256, inDouble256.length);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex032(Complex32 state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex032Blocked(Complex32Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex032SIMD(Complex32 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex032SIMDBlocked(Complex32Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex064(Complex64 state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex064Blocked(Complex64Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex064SIMD(Complex64 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex064SIMDBlocked(Complex64Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128Blocked(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMD(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop004(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(4);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop008(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(8);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop016(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(16);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop032(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(32);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop064(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(64);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDLoop128(Complex128 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(128);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlocked(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop004(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(4);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop008(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(8);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop016(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(16);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop032(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(32);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop064(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(64);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex128SIMDBlockedLoop128(Complex128Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.setMinSIMDLoopLength(128);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex256(Complex256 state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex256Blocked(Complex256Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(false);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex256SIMD(Complex256 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex256SIMDBlocked(Complex256Blocked state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 1);
    blackhole.consume(state.in);
  }

}
