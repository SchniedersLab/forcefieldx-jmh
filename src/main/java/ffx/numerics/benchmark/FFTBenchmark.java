
package ffx.numerics.benchmark;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
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
  private final static int measurementIterations = 3;
  /**
   * Each measurement iteration will run for this many seconds.
   */
  private final static int measurementTime = 1;

  public static final int n = 128;
  public static final int[] factors = {2, 2, 2, 2, 2, 2, 2};
  public static final double[] in64 = new double[n * 2];
  public static final float[] in32 = new float[n * 2];
  public static int sign = 1;
  public static int pass2InnerLoopCycles = n / 2;

  static {
    Random random = new Random(1);
    for (int i = 0; i < n; i++) {
      in32[i * 2] = random.nextFloat();
      in64[i * 2] = random.nextDouble();
    }
  }

  private static final VectorSpecies<Double> VSPEC_F64 = DoubleVector.SPECIES_PREFERRED;
  private static final VectorMask<Double> pass2Mask64;
  private static final VectorShuffle<Double> pass2Shuffle64;

  static {
    boolean[] negateMask;
    int[] shuffleMask;
    if (VSPEC_F64 == DoubleVector.SPECIES_512) {
      negateMask = new boolean[]{false, true, false, true, false, true, false, true};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6};
    } else if (VSPEC_F64 == DoubleVector.SPECIES_256) {
      negateMask = new boolean[]{false, true, false, true};
      shuffleMask = new int[]{1, 0, 3, 2};
    } else {
      negateMask = new boolean[]{false, true};
      shuffleMask = new int[]{1, 0};
    }
    pass2Mask64 = VectorMask.fromArray(VSPEC_F64, negateMask, 0);
    pass2Shuffle64 = VectorShuffle.fromArray(VSPEC_F64, shuffleMask, 0);
    System.out.println("\nDoubleVector.SPECIES_PREFERRED: " + VSPEC_F64);
    System.out.println("Scalar Inner loop cycles: " + pass2InnerLoopCycles);
    System.out.println("SIMD Inner loop cycles:   " + pass2InnerLoopCycles / (VSPEC_F64.length() / 2));
  }

  private static final VectorSpecies<Float> VSPEC_F32 = FloatVector.SPECIES_PREFERRED;
  private static final VectorMask<Float> pass2Mask32;
  private static final VectorShuffle<Float> pass2Shuffle32;

  static {
    boolean[] negateMask;
    int[] shuffleMask;
    if (VSPEC_F32 == FloatVector.SPECIES_512) {
      negateMask = new boolean[]{
          false, true, false, true, false, true, false, true,
          false, true, false, true, false, true, false, true};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    } else if (VSPEC_F32 == FloatVector.SPECIES_256) {
      negateMask = new boolean[]{false, true, false, true, false, true, false, true};
      shuffleMask = new int[]{1, 0, 3, 2, 5, 4, 7, 6};
    } else if (VSPEC_F32 == FloatVector.SPECIES_128) {
      negateMask = new boolean[]{false, true, false, true};
      shuffleMask = new int[]{1, 0, 3, 2};
    } else {
      negateMask = new boolean[]{false, true};
      shuffleMask = new int[]{1, 0};
    }
    pass2Mask32 = VectorMask.fromArray(VSPEC_F32, negateMask, 0);
    pass2Shuffle32 = VectorShuffle.fromArray(VSPEC_F32, shuffleMask, 0);
    System.out.println("\nFloatVector.SPECIES_PREFERRED: " + VSPEC_F32);
    System.out.println("Scalar Inner loop cycles: " + pass2InnerLoopCycles);
    System.out.println("SIMD Inner loop cycles:   " + pass2InnerLoopCycles / (VSPEC_F32.length() / 2));
  }

  @State(Scope.Thread)
  public static class FFTState_PassData64 {
    static final double[][][] twiddles = wavetable();
    double[] in = Arrays.copyOf(in64, in64.length);
    double[] out = new double[n * 2];
    PassData64 passData64 = new PassData64(in, 0, out, 0);
  }

  @State(Scope.Thread)
  public static class FFTState_PassData32 {
    static final float[][][] twiddles = wavetableFloat();
    float[] in = Arrays.copyOf(in32, in32.length);
    float[] out = new float[n * 2];
    PassData32 passData32 = new PassData32(in, 0, out, 0);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleVectorFFTPass2(FFTState_PassData64 state, Blackhole blackhole) {
    int product = n;
    doublePass2(product, state.passData64, FFTState_PassData64.twiddles[factors.length - 1]);
    blackhole.consume(state.passData64.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleVectorFFTPass2SIMD(FFTState_PassData64 state, Blackhole blackhole) {
    int product = n;
    doublePass2SIMD(product, state.passData64, FFTState_PassData64.twiddles[factors.length - 1]);
    blackhole.consume(state.passData64.out);
  }

  private void doublePass2(int product, PassData64 passData64, double[][] twiddles) {
    final double[] data = passData64.in;
    final double[] ret = passData64.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData64.inOffset;
    int j = passData64.outOffset;
    for (int k = 0; k < q; k++, j += dj) {
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

  private void doublePass2SIMD(int product, PassData64 passData64, double[][] twiddles) {
    final double[] data = passData64.in;
    final double[] ret = passData64.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData64.inOffset;
    int j = passData64.outOffset;
    final int dataInc = VSPEC_F64.length();
    final int k1inc = dataInc / 2;
    for (int k = 0; k < q; k++, j += dj) {
      final double[] twids = twiddles[k];
      DoubleVector
          w_r = DoubleVector.broadcast(VSPEC_F64, twids[0]),
          w_i = DoubleVector.broadcast(VSPEC_F64, -sign * twids[1]).mul(-1.0, pass2Mask64);
      for (int k1 = 0; k1 < product_1; k1 += k1inc, i += dataInc, j += dataInc) {
        DoubleVector
            z0 = DoubleVector.fromArray(VSPEC_F64, data, i),
            z1 = DoubleVector.fromArray(VSPEC_F64, data, i + di),
            sum = z0.add(z1),
            x = z0.sub(z1),
            sum2 = x.fma(w_r, x.mul(w_i).rearrange(pass2Shuffle64));
        sum.intoArray(ret, j);
        sum2.intoArray(ret, j + dj);
      }
    }
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void floatVectorFFTPass2(FFTState_PassData32 state, Blackhole blackhole) {
    int product = n;
    floatPass2(product, state.passData32, FFTState_PassData32.twiddles[factors.length - 1]);
    blackhole.consume(state.passData32.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void floatVectorFFTPass2SIMD(FFTState_PassData32 state, Blackhole blackhole) {
    int product = n;
    floatPass2SIMD(product, state.passData32, FFTState_PassData32.twiddles[factors.length - 1]);
    blackhole.consume(state.passData32.out);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void floatVectorFFTPass2SIMDUnroll(FFTState_PassData32 state, Blackhole blackhole) {
    int product = n;
    floatPass2SIMDUnroll(product, state.passData32, FFTState_PassData32.twiddles[factors.length - 1]);
    blackhole.consume(state.passData32.out);
  }

  private void floatPass2(int product, PassData32 passData64, float[][] twiddles) {
    final float[] data = passData64.in;
    final float[] ret = passData64.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData64.inOffset;
    int j = passData64.outOffset;
    for (int k = 0; k < q; k++, j += dj) {
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

  private void floatPass2SIMD(int product, PassData32 passData64, float[][] twiddles) {
    final float[] data = passData64.in;
    final float[] ret = passData64.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData64.inOffset;
    int j = passData64.outOffset;
    final int dataInc = VSPEC_F32.length();
    final int k1inc = dataInc / 2;
    for (int k = 0; k < q; k++, j += dj) {
      final float[] twids = twiddles[k];
      FloatVector
          w_r = FloatVector.broadcast(VSPEC_F32, twids[0]),
          w_i = FloatVector.broadcast(VSPEC_F32, -sign * twids[1]).mul(-1.0f, pass2Mask32);
      for (int k1 = 0; k1 < product_1; k1 += k1inc, i += dataInc, j += dataInc) {
        FloatVector
            z0 = FloatVector.fromArray(VSPEC_F32, data, i),
            z1 = FloatVector.fromArray(VSPEC_F32, data, i + di),
            sum = z0.add(z1),
            x = z0.sub(z1),
            sum2 = x.fma(w_r, x.mul(w_i).rearrange(pass2Shuffle32));
        sum.intoArray(ret, j);
        sum2.intoArray(ret, j + dj);
      }
    }
  }

  private void floatPass2SIMDUnroll(int product, PassData32 passData64, float[][] twiddles) {
    final float[] data = passData64.in;
    final float[] ret = passData64.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData64.inOffset;
    int j = passData64.outOffset;
    final int dataInc = VSPEC_F32.length();
    final int dataInc2 = dataInc * 2;
    final int k1inc = dataInc;
    for (int k = 0; k < q; k++, j += dj) {
      final float[] twids = twiddles[k];
      FloatVector
          w_r = FloatVector.broadcast(VSPEC_F32, twids[0]),
          w_i = FloatVector.broadcast(VSPEC_F32, -sign * twids[1]).mul(-1.0f, pass2Mask32);
      for (int k1 = 0; k1 < product_1; k1 += k1inc, i += dataInc, j += dataInc) {
        FloatVector
            z0 = FloatVector.fromArray(VSPEC_F32, data, i),
            z1 = FloatVector.fromArray(VSPEC_F32, data, i + di),
            sum = z0.add(z1),
            x = z0.sub(z1),
            sum2 = x.fma(w_r, x.mul(w_i).rearrange(pass2Shuffle32));
        sum.intoArray(ret, j);
        sum2.intoArray(ret, j + dj);

        // Second iteration.
        i += dataInc;
        j += dataInc;
        z0 = FloatVector.fromArray(VSPEC_F32, data, i);
        z1 = FloatVector.fromArray(VSPEC_F32, data, i + di);
        sum = z0.add(z1);
        x = z0.sub(z1);
        sum2 = x.fma(w_r, x.mul(w_i).rearrange(pass2Shuffle32));
        sum.intoArray(ret, j);
        sum2.intoArray(ret, j + dj);
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
  private record PassData64(double[] in, int inOffset, double[] out, int outOffset) {
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
  private record PassData32(float[] in, int inOffset, float[] out, int outOffset) {
    // Empty.
  }

  /**
   * Compute twiddle factors. These are trigonometric constants that depend on the factoring of n.
   *
   * @return twiddle factors.
   */
  private static double[][][] wavetable() {
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
