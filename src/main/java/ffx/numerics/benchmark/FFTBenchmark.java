
package ffx.numerics.benchmark;

import ffx.numerics.fft.Complex;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.math3.util.FastMath;
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

import static java.lang.Math.fma;
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

  public static final int n = 128;
  public static final int[] factors = {2, 2, 2, 2, 2, 2, 2};
  public static final double[] inDouble = new double[n * 2];
  public static final float[] inFloat = new float[n * 2];
  public static int sign = 1;

  public static final double[] inDouble32 = new double[32 * 2];
  public static final double[] inDouble64 = new double[64 * 2];
  public static final double[] inDouble128 = new double[128 * 2];
  public static final double[] inDouble256 = new double[256 * 2];

  // Initialize the input arrays with random values.
  static {
    Random random = new Random(1);
    for (int i = 0; i < n; i++) {
      inFloat[i * 2] = random.nextFloat();
      inDouble[i * 2] = random.nextDouble();
    }

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

  private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
  private static final DoubleVector negateIm;
  private static final VectorShuffle<Double> pass2ShuffleDouble;

  static {
    double[] negate;
    int[] shuffleMask;
    if (DOUBLE_SPECIES == DoubleVector.SPECIES_512) {
      negate = new double[]{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6};
    } else if (DOUBLE_SPECIES == DoubleVector.SPECIES_256) {
      negate = new double[]{1.0, -1.0, 1.0, -1.0};
      shuffleMask = new int[]{1, 0, 3, 2};
    } else {
      negate = new double[]{1.0, -1.0};
      shuffleMask = new int[]{1, 0};
    }
    negateIm = DoubleVector.fromArray(DOUBLE_SPECIES, negate, 0);
    pass2ShuffleDouble = VectorShuffle.fromArray(DOUBLE_SPECIES, shuffleMask, 0);
    System.out.println("\nDoubleVector.SPECIES_PREFERRED: " + DOUBLE_SPECIES);
  }

  private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final FloatVector negateImFloat;
  private static final VectorShuffle<Float> pass2ShuffleFloat;

  static {
    float[] negate;
    int[] shuffleMask;
    if (FLOAT_SPECIES == FloatVector.SPECIES_512) {
      negate = new float[]{
          1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
          1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    } else if (FLOAT_SPECIES == FloatVector.SPECIES_256) {
      negate = new float[]{1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6};
    } else if (FLOAT_SPECIES == FloatVector.SPECIES_128) {
      negate = new float[]{1.0f, -1.0f, 1.0f, -1.0f};
      shuffleMask = new int[]{1, 0, 3, 2};
    } else {
      negate = new float[]{1.0f, -1.0f, 1.0f, -1.0f, 1.0f};
      shuffleMask = new int[]{1, 0};
    }
    negateImFloat = FloatVector.fromArray(FLOAT_SPECIES, negate, 0);
    pass2ShuffleFloat = VectorShuffle.fromArray(FLOAT_SPECIES, shuffleMask, 0);
    System.out.println("\nFloatVector.SPECIES_PREFERRED: " + FLOAT_SPECIES);
  }

  @State(Scope.Thread)
  public static class FFTState_PassData64 {
    static final double[][][] twiddles = wavetableDouble();
    double[] in = Arrays.copyOf(inDouble, inDouble.length);
    double[] out = new double[n * 2];
    PassDataDouble passDataDouble = new PassDataDouble(in, 0, out, 0);
  }

  @State(Scope.Thread)
  public static class FFTState_PassData32 {
    static final float[][][] twiddles = wavetableFloat();
    float[] in = Arrays.copyOf(inFloat, inFloat.length);
    float[] out = new float[n * 2];
    PassDataFloat passDataFloat = new PassDataFloat(in, 0, out, 0);
  }

  @State(Scope.Thread)
  public static class Complex32 {
    Complex complex = new Complex(32);
    double[] in = Arrays.copyOf(inDouble32, inDouble32.length);
  }

  @State(Scope.Thread)
  public static class Complex64 {
    Complex complex = new Complex(64);
    double[] in = Arrays.copyOf(inDouble64, inDouble64.length);
  }

  @State(Scope.Thread)
  public static class Complex128 {
    Complex complex = new Complex(128);
    double[] in = Arrays.copyOf(inDouble128, inDouble128.length);
  }

  @State(Scope.Thread)
  public static class Complex256 {
    Complex complex = new Complex(256);
    double[] in = Arrays.copyOf(inDouble256, inDouble256.length);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Pass2DoubleVectorFFT(FFTState_PassData64 state, Blackhole blackhole) {
    int product = 2;
    for (int i = 1; i < factors.length; i++) {
      int factor = factors[i];
      product *= factor;
      doublePass2(product, state.passDataDouble, FFTState_PassData64.twiddles[i]);
    }
    blackhole.consume(state.passDataDouble.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Pass2DoubleVectorFFTSIMD(FFTState_PassData64 state, Blackhole blackhole) {
    int product = 2;
    for (int i = 1; i < factors.length; i++) {
      int factor = factors[i];
      product *= factor;
      doublePass2SIMD(product, state.passDataDouble, FFTState_PassData64.twiddles[i]);
    }
    blackhole.consume(state.passDataDouble.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Pass2DoubleVectorFFTSIMDBlocked(FFTState_PassData64 state, Blackhole blackhole) {
    int product = 2;
    // From interleaved to packed.
//    double[] in = state.in;
//    double[] out = state.out;
//    int len = in.length;
//    for (int iin = 0, kout = 0; iin < len; iin+=2, kout += 1) {
//      double re = in[iin];
//      double im = in[iin + 1];
//      out[kout] = re;
//      out[kout + n] = im;
//    }
//    state.in = out;
//    state.out = in;
    for (int i = 1; i < factors.length; i++) {
      int factor = factors[i];
      product *= factor;
      doublePass2SIMDBlocked(product, state.passDataDouble, FFTState_PassData64.twiddles[i]);
    }
    blackhole.consume(state.passDataDouble.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Pass2FloatVectorFFT(FFTState_PassData32 state, Blackhole blackhole) {
    int product = 2;
    for (int i = 1; i < factors.length; i++) {
      int factor = factors[i];
      product *= factor;
      floatPass2(product, state.passDataFloat, FFTState_PassData32.twiddles[i]);
    }
    blackhole.consume(state.passDataFloat.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {
      "--add-modules=jdk.incubator.vector"
      // "-XX:+UnlockDiagnosticVMOptions",
      // "-XX:+LogCompilation",
      // "-XX:CompileCommand=print ffx.numerics.benchmark::floatPass2SIMD"
  })
  public void Pass2FloatVectorFFTSIMD(FFTState_PassData32 state, Blackhole blackhole) {
    int product = 2;
    for (int i = 1; i < factors.length; i++) {
      int factor = factors[i];
      product *= factor;
      floatPass2SIMD(product, state.passDataFloat, FFTState_PassData32.twiddles[i]);
    }
    blackhole.consume(state.passDataFloat.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void Complex32(Complex32 state, Blackhole blackhole) {
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
  public void Complex32SIMD(Complex32 state, Blackhole blackhole) {
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
  public void Complex64(Complex64 state, Blackhole blackhole) {
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
  public void Complex64SIMD(Complex64 state, Blackhole blackhole) {
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
  public void Complex256SIMD(Complex256 state, Blackhole blackhole) {
    state.complex.setUseSIMD(true);
    state.complex.fft(state.in, 0, 2);
    blackhole.consume(state.in);
  }

  private void doublePass2(int product, PassDataDouble passDataDouble, double[][] twiddles) {
    final double[] data = passDataDouble.in;
    final double[] ret = passDataDouble.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passDataDouble.inOffset;
    int j = passDataDouble.outOffset;

    // First iteration has no twiddle factors.
    for (int k1 = 0; k1 < product_1; k1++, i += 2, j += 2) {
      final double z0_r = data[i];
      final double z0_i = data[i + 1];
      final int idi = i + di;
      final double z1_r = data[idi];
      final double z1_i = data[idi + 1];
      ret[j] = z0_r + z1_r;
      ret[j + 1] = z0_i + z1_i;
      final int jdj = j + dj;
      ret[jdj] = z0_r - z1_r;
      ret[jdj + 1] = z0_i - z1_i;
    }

    j += dj;
    for (int k = 1; k < q; k++, j += dj) {
      final double[] twids = twiddles[k];
      final double w_r = twids[0];
      final double w_i = -sign * twids[1];
      for (int k1 = 0; k1 < product_1; k1++, i += 2, j += 2) {
        final double z0_r = data[i];
        final double z0_i = data[i + 1];
        final int idi = i + di;
        final double z1_r = data[idi];
        final double z1_i = data[idi + 1];
        ret[j] = z0_r + z1_r;
        ret[j + 1] = z0_i + z1_i;
        final double x_r = z0_r - z1_r;
        final double x_i = z0_i - z1_i;
        final int jdj = j + dj;
        ret[jdj] = fma(w_r, x_r, -w_i * x_i);
        ret[jdj + 1] = fma(w_r, x_i, w_i * x_r);
      }
    }
  }

  private void doublePass2SIMD(int product, PassDataDouble passDataDouble, double[][] twiddles) {
    final double[] data = passDataDouble.in;
    final double[] ret = passDataDouble.out;
    final int factor = 2;
    final int m = n / factor;
    final int outerLoopLimit = n / product;
    final int innerLoopLimit = product / factor;
    final int dataInc = DOUBLE_SPECIES.length();
    final int k1inc = dataInc / 2;

    if (innerLoopLimit % k1inc != 0) {
      // System.out.printf("Scalar %d product=%d innerLoopLimit=%d increment=%d%n",
      // factor, product, innerLoopLimit, LOOP_INCREMENT);
      doublePass2(product, passDataDouble, twiddles);
      return;
    }

    final int di = 2 * m;
    final int dj = 2 * innerLoopLimit;
    int i = passDataDouble.inOffset;
    int j = passDataDouble.outOffset;


    // First iteration has no twiddle factors.
    for (int k1 = 0; k1 < innerLoopLimit; k1 += k1inc, i += dataInc, j += dataInc) {
      final int idi = i + di;
      final int jdj = j + dj;
      DoubleVector z0 = DoubleVector.fromArray(DOUBLE_SPECIES, data, i);
      DoubleVector z1 = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi);
      z0.add(z1).intoArray(ret, j);
      z0.sub(z1).intoArray(ret, jdj);
    }

    j += dj;
    for (int k = 1; k < outerLoopLimit; k++, j += dj) {
      final double[] twids = twiddles[k];
      DoubleVector
          w_r = DoubleVector.broadcast(DOUBLE_SPECIES, twids[0]),
          w_i = DoubleVector.broadcast(DOUBLE_SPECIES, -sign * twids[1]).mul(negateIm);
      for (int k1 = 0; k1 < innerLoopLimit; k1 += k1inc, i += dataInc, j += dataInc) {
        final int idi = i + di;
        final int jdj = j + dj;
        DoubleVector z0 = DoubleVector.fromArray(DOUBLE_SPECIES, data, i);
        DoubleVector z1 = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi);
        DoubleVector x = z0.sub(z1);
        DoubleVector xw_r = x.mul(w_r);
        DoubleVector xw_i = x.mul(w_i);
        xw_i = xw_i.rearrange(pass2ShuffleDouble);
        DoubleVector sum = z0.add(z1);
        sum.intoArray(ret, j);
        DoubleVector sum2 = xw_r.add(xw_i);
        sum2.intoArray(ret, jdj);
      }
    }
  }

  private void doublePass2SIMDBlocked(int product, PassDataDouble passDataDouble, double[][] twiddles) {
    final double[] data = passDataDouble.in;
    final double[] ret = passDataDouble.out;
    final int factor = 2;
    final int innerLoopLimit = product / factor;

    // Number of complex number processed in one pass.
    final int k1inc = DOUBLE_SPECIES.length();
    // Collect a vector of set of real and imaginary parts each inner loop cycle.
    final int dataInc = DOUBLE_SPECIES.length();

    if (innerLoopLimit % k1inc != 0) {
      // System.out.printf("Scalar %d product=%d innerLoopLimit=%d increment=%d%n",
      // factor, product, innerLoopLimit, LOOP_INCREMENT);
      doublePass2(product, passDataDouble, twiddles);
      return;
    }

    final int m = n / factor;
    final int outerLoopLimit = n / product;
    final int di = 2 * m;
    final int dj = 2 * innerLoopLimit;
    int i = passDataDouble.inOffset;
    int j = passDataDouble.outOffset;
    int imOffset = n / 2;

    // First iteration has no twiddle factors.
    for (int k1 = 0; k1 < innerLoopLimit; k1 += k1inc, i += dataInc, j += dataInc) {
      DoubleVector z0_r = DoubleVector.fromArray(DOUBLE_SPECIES, data, i);
      DoubleVector z0_i = DoubleVector.fromArray(DOUBLE_SPECIES, data, i + imOffset);
      final int idi = i + di;
      DoubleVector z1_r = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi);
      DoubleVector z1_i = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi + imOffset);
      z0_r.add(z1_r).intoArray(ret, j);
      z0_i.add(z1_i).intoArray(ret, j + imOffset);
      final int jdj = j + dj;
      z0_r.sub(z1_r).intoArray(ret, jdj);
      z0_i.sub(z1_i).intoArray(ret, jdj + imOffset);
    }

    j += dj;
    for (int k = 1; k < outerLoopLimit; k++, j += dj) {
      final double[] twids = twiddles[k];
      DoubleVector
          w_r = DoubleVector.broadcast(DOUBLE_SPECIES, twids[0]),
          w_i = DoubleVector.broadcast(DOUBLE_SPECIES, -sign * twids[1]);
      for (int k1 = 0; k1 < innerLoopLimit; k1 += dataInc, i += dataInc, j += dataInc) {
        DoubleVector z0_r = DoubleVector.fromArray(DOUBLE_SPECIES, data, i);
        DoubleVector z0_i = DoubleVector.fromArray(DOUBLE_SPECIES, data, i + imOffset);
        final int idi = i + di;
        DoubleVector z1_r = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi);
        DoubleVector z1_i = DoubleVector.fromArray(DOUBLE_SPECIES, data, idi + imOffset);
        z0_r.add(z1_r).intoArray(ret, j);
        z0_i.add(z1_i).intoArray(ret, j + imOffset);
        DoubleVector x_r = z0_r.sub(z1_r);
        DoubleVector x_i = z0_i.sub(z1_i);
        final int jdj = j + dj;
        w_r.fma(x_r, w_i.neg().mul(x_i)).intoArray(ret, jdj);
        w_r.fma(x_i, w_i.mul(x_r)).intoArray(ret, jdj + imOffset);
      }
    }
  }

  private void floatPass2(int product, PassDataFloat passData32, float[][] twiddles) {
    final float[] data = passData32.in;
    final float[] ret = passData32.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData32.inOffset;
    int j = passData32.outOffset;

    for (int k1 = 0; k1 < product_1; k1++, i += 2, j += 2) {
      final float z0_r = data[i];
      final float z0_i = data[i + 1];
      final int idi = i + di;
      final float z1_r = data[idi];
      final float z1_i = data[idi + 1];
      ret[j] = z0_r + z1_r;
      ret[j + 1] = z0_i + z1_i;
      final int jdj = j + dj;
      ret[jdj] = z0_r - z1_r;
      ret[jdj + 1] = z0_i - z1_i;
    }

    j += dj;
    for (int k = 1; k < q; k++, j += dj) {
      final float[] twids = twiddles[k];
      final float w_r = twids[0];
      final float w_i = -sign * twids[1];
      for (int k1 = 0; k1 < product_1; k1++, i += 2, j += 2) {
        final float z0_r = data[i];
        final float z0_i = data[i + 1];
        final int idi = i + di;
        final float z1_r = data[idi];
        final float z1_i = data[idi + 1];
        ret[j] = z0_r + z1_r;
        ret[j + 1] = z0_i + z1_i;
        final float x_r = z0_r - z1_r;
        final float x_i = z0_i - z1_i;
        final int jdj = j + dj;
        ret[jdj] = fma(w_r, x_r, -w_i * x_i);
        ret[jdj + 1] = fma(w_r, x_i, w_i * x_r);
      }
    }
  }

  private void floatPass2SIMD(int product, PassDataFloat passData32, float[][] twiddles) {
    final float[] data = passData32.in;
    final float[] ret = passData32.out;
    final int factor = 2;
    final int m = n / factor;
    final int outerLoopLimit = n / product;
    final int innerLoopLimit = product / factor;
    final int dataInc = FLOAT_SPECIES.length();
    final int k1inc = dataInc / 2;

    if (innerLoopLimit % k1inc != 0) {
      System.out.printf("Scalar %d product=%d inner loop limit=%d loop increment=%d%n",
          factor, product, innerLoopLimit, k1inc);
      floatPass2(product, passData32, twiddles);
      return;
    }

    final int di = 2 * m;
    final int dj = 2 * innerLoopLimit;
    int i = passData32.inOffset;
    int j = passData32.outOffset;

    // First iteration has no twiddle factors.
    for (int k1 = 0; k1 < innerLoopLimit; k1 += k1inc, i += dataInc, j += dataInc) {
      final int idi = i + di;
      final int jdj = j + dj;
      FloatVector z0 = FloatVector.fromArray(FLOAT_SPECIES, data, i);
      FloatVector z1 = FloatVector.fromArray(FLOAT_SPECIES, data, idi);
      z0.add(z1).intoArray(ret, j);
      z0.sub(z1).intoArray(ret, jdj);
    }

    j += dj;
    for (int k = 1; k < outerLoopLimit; k++, j += dj) {
      final float[] twids = twiddles[k];
      FloatVector w_r = FloatVector.broadcast(FLOAT_SPECIES, twids[0]);
      FloatVector w_i = FloatVector.broadcast(FLOAT_SPECIES, -sign * twids[1]).mul(negateImFloat);
      for (int k1 = 0; k1 < innerLoopLimit; k1 += k1inc, i += dataInc, j += dataInc) {
        final int idi = i + di;
        final int jdj = j + dj;
        FloatVector z0 = FloatVector.fromArray(FLOAT_SPECIES, data, i);
        FloatVector z1 = FloatVector.fromArray(FLOAT_SPECIES, data, idi);
        FloatVector sum = z0.add(z1);
        sum.intoArray(ret, j);
        FloatVector x = z0.sub(z1);
        FloatVector sum2 = x.fma(w_r, x.mul(w_i).rearrange(pass2ShuffleFloat));
        sum2.intoArray(ret, jdj);
      }
    }
  }

  /**
   * References to the input and output data arrays.
   *
   * @param in        Input data for the current pass.
   * @param inOffset  Offset into the input data.
   * @param out       Output data for the current pass.
   * @param outOffset Offset into output array.
   */
  private record PassDataDouble(double[] in, int inOffset, double[] out, int outOffset) {
    // Empty.
  }

  /**
   * References to the input and output data arrays.
   *
   * @param in        Input data for the current pass.
   * @param inOffset  Offset into the input data.
   * @param out       Output data for the current pass.
   * @param outOffset Offset into output array.
   */
  private record PassDataFloat(float[] in, int inOffset, float[] out, int outOffset) {
    // Empty.
  }

  /**
   * Compute twiddle factors. These are trigonometric constants that depend on the factoring of n.
   *
   * @return twiddle factors.
   */
  private static double[][][] wavetableDouble() {
    if (n < 2) {
      return null;
    }
    final double d_theta = -2.0 * FastMath.PI / n;
    final double[][][] ret = new double[factors.length][][];
    int product = 1;
    for (int i = 0; i < factors.length; i++) {
      int factor = factors[i];
      int product_1 = product;
      product *= factor;
      final int q = n / product;
      ret[i] = new double[q + 1][2 * (factor - 1)];
      final double[][] twid = ret[i];
      for (int j = 0; j < factor - 1; j++) {
        twid[0][2 * j] = 1.0;
        twid[0][2 * j + 1] = 0.0;
      }
      for (int k = 1; k <= q; k++) {
        int m = 0;
        for (int j = 0; j < factor - 1; j++) {
          m += k * product_1;
          m %= n;
          final double theta = d_theta * m;
          twid[k][2 * j] = FastMath.cos(theta);
          twid[k][2 * j + 1] = FastMath.sin(theta);
        }
      }
    }
    return ret;
  }

  /**
   * Compute twiddle factors. These are trigonometric constants that depend on the factoring of n.
   *
   * @return twiddle factors.
   */
  private static float[][][] wavetableFloat() {
    if (n < 2) {
      return null;
    }
    final double d_theta = -2.0 * FastMath.PI / n;
    final float[][][] ret = new float[factors.length][][];
    int product = 1;
    for (int i = 0; i < factors.length; i++) {
      int factor = factors[i];
      int product_1 = product;
      product *= factor;
      final int q = n / product;
      ret[i] = new float[q + 1][2 * (factor - 1)];
      final float[][] twid = ret[i];
      for (int j = 0; j < factor - 1; j++) {
        twid[0][2 * j] = 1.0f;
        twid[0][2 * j + 1] = 0.0f;
      }
      for (int k = 1; k <= q; k++) {
        int m = 0;
        for (int j = 0; j < factor - 1; j++) {
          m += k * product_1;
          m %= n;
          final double theta = d_theta * m;
          twid[k][2 * j] = (float) FastMath.cos(theta);
          twid[k][2 * j + 1] = (float) FastMath.sin(theta);
        }
      }
    }
    return ret;
  }

}
