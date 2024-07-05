
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

import static java.lang.Integer.numberOfTrailingZeros;
import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.fma;
import static java.lang.Math.random;
import static java.lang.Math.sin;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static jdk.incubator.vector.DoubleVector.SPECIES_128;
import static jdk.incubator.vector.DoubleVector.SPECIES_256;
import static jdk.incubator.vector.DoubleVector.SPECIES_512;
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

  private static final VectorSpecies<Float> VSPEC_F32 = FloatVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Double> VSPEC_F64 = DoubleVector.SPECIES_PREFERRED;

  private static boolean hasOneBitOrNone(int n) {
    return (n & n - 1) == 0;
  }

  public static int len() {
    return VSPEC_F32.length();
  }

  public static int len64() {
    return VSPEC_F64.length();
  }

  public static final int n = 128;
  public static final int[] factors = {2, 2, 2, 2, 2, 2, 2};
  public static int sign = 1;

  /**
   * Pre-computed primitive roots of unity.
   */
  private static final float[] PROUS = new float[2 * len()];

  private static final double[] PROUS_64 = new double[2 * len64()];

  static {
    int vLen = VSPEC_F32.length();
    for (int i = 0; i < vLen; i++) {
      var x = -2 * i * PI / vLen;
      PROUS[2 * i] = (float) cos(x);
      PROUS[2 * i + 1] = (float) sin(x);
    }

    vLen = VSPEC_F64.length();
    for (int i = 0; i < vLen; i++) {
      var x = -2 * i * PI / vLen;
      PROUS_64[2 * i] = (float) cos(x);
      PROUS_64[2 * i + 1] = (float) sin(x);
    }

  }

  @SuppressWarnings("unchecked")
  private static final VectorShuffle<Float>[] INDICES = new VectorShuffle[numberOfTrailingZeros(len()) * 2];
  private static final FloatVector[] FACTORS = new FloatVector[numberOfTrailingZeros(len()) * 2];

  @SuppressWarnings("unchecked")
  private static final VectorShuffle<Double>[] INDICES_64 = new VectorShuffle[numberOfTrailingZeros(len64()) * 2];
  private static final DoubleVector[] FACTORS_64 = new DoubleVector[numberOfTrailingZeros(len64()) * 2];

  static {
    int vLen = VSPEC_F32.length();
    var shuffle = new int[vLen * 2];
    var roots = new float[vLen * 2];

    for (int k = 0, b = 1; b < vLen; b <<= 1, k += 2) {
      for (int i = 0; i < vLen; i += b) {
        var w0 = -PI * 2 / (b << 1);
        int wi = 0;
        for (int end = i | b; i < end; i++) {
          int j = i | b;
          double x = w0 * wi++;
          roots[j] = -(roots[i] = (float) cos(x));
          roots[j + vLen] = -(roots[i + vLen] = (float) sin(x));
          shuffle[i] = shuffle[j] = i;
          shuffle[i + vLen] = shuffle[j + vLen] = j;
        }
      }
      INDICES[k] = VectorShuffle.fromArray(VSPEC_F32, shuffle, 0);
      INDICES[k + 1] = VectorShuffle.fromArray(VSPEC_F32, shuffle, vLen);
      FACTORS[k] = FloatVector.fromArray(VSPEC_F32, roots, 0);
      FACTORS[k + 1] = FloatVector.fromArray(VSPEC_F32, roots, vLen);
    }

    int vLen64 = VSPEC_F64.length();
    var shuffle64 = new int[vLen64 * 2];
    var roots64 = new double[vLen64 * 2];

    for (int k = 0, b = 1; b < vLen64; b <<= 1, k += 2) {
      for (int i = 0; i < vLen64; i += b) {
        var w0 = -PI * 2 / (b << 1);
        int wi = 0;
        for (int end = i | b; i < end; i++) {
          int j = i | b;
          double x = w0 * wi++;
          roots64[j] = -(roots64[i] = cos(x));
          roots64[j + vLen64] = -(roots64[i + vLen64] = sin(x));
          shuffle64[i] = shuffle64[j] = i;
          shuffle64[i + vLen64] = shuffle64[j + vLen64] = j;
        }
      }
      INDICES_64[k] = VectorShuffle.fromArray(VSPEC_F64, shuffle64, 0);
      INDICES_64[k + 1] = VectorShuffle.fromArray(VSPEC_F64, shuffle64, vLen64);
      FACTORS_64[k] = DoubleVector.fromArray(VSPEC_F64, roots64, 0);
      FACTORS_64[k + 1] = DoubleVector.fromArray(VSPEC_F64, roots64, vLen64);
    }

  }

  private static final VectorMask<Double> mask;
  private static final VectorShuffle<Double> shuffle;

  static {
    if (VSPEC_F64 == SPECIES_512) {
      boolean[] negateMask = {false, true, false, true, false, true, false, true};
      mask = VectorMask.fromArray(SPECIES_512, negateMask, 0);
      int[] shuffleMask = {1, 0, 3, 2, 5, 4, 7, 6};
      shuffle = VectorShuffle.fromArray(SPECIES_512, shuffleMask, 0);
    } else if (VSPEC_F64 == SPECIES_256) {
      boolean[] negateMask = {false, true, false, true};
      mask = VectorMask.fromArray(SPECIES_256, negateMask, 0);
      int[] shuffleMask = {1, 0, 3, 2};
      shuffle = VectorShuffle.fromArray(SPECIES_256, shuffleMask, 0);
    } else {
      boolean[] negateMask = {false, true};
      mask = VectorMask.fromArray(SPECIES_128, negateMask, 0);
      int[] shuffleMask = {1, 0};
      shuffle = VectorShuffle.fromArray(SPECIES_128, shuffleMask, 0);
    }
  }

  @State(Scope.Thread)
  public static class FFTState {

    int len = len();
    float[] re = new float[len];
    float[] im = new float[len];

    public FFTState() {
      for (int i = 0; i < len; i++) {
        re[i] = (float) random();
      }
    }
  }

  @State(Scope.Thread)
  public static class FFTState_64 {

    int len = len64();
    double[] re = new double[len];
    double[] im = new double[len];

    public FFTState_64() {
      for (int i = 0; i < len; i++) {
        re[i] = random();
      }
    }
  }

  @State(Scope.Thread)
  public static class FFTState_PassData {
    double[] in = new double[n * 2];
    double[] out = new double[n * 2];
    PassData passData = new PassData(in, 0, out, 0);
    double[][][] twiddles = wavetable();
  }


  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void floatScalarFFT(FFTState state, Blackhole blackhole) {
    fftV1(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void floatVectorFFT(FFTState state, Blackhole blackhole) {
    fftV2(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleScalarFFT(FFTState_64 state, Blackhole blackhole) {
    fftV1_64(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleVectorFFT(FFTState_64 state, Blackhole blackhole) {
    fftV2_64(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleVectorFFTPass2(FFTState_PassData state, Blackhole blackhole) {
    int product = 2;
    pass2(product, state.passData, state.twiddles[0]);
    blackhole.consume(state.passData.out);
  }
  
  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void doubleVectorFFTPass2SIMD(FFTState_PassData state, Blackhole blackhole) {
    int product = 2;
    pass2SIMD(product, state.passData, state.twiddles[0]);
    blackhole.consume(state.passData.out);
  }


  public static void fftV1(float[] re, float[] im) {
    int n = re.length;
    if (n != im.length || n != len() || !hasOneBitOrNone(n)) {
      throw new IllegalArgumentException();
    }

    for (int dr = n, b = 1; b < n; b <<= 1, dr >>= 1) {
      for (int i = 0; i < n; i += b) {
        for (int root = 0; root < b; root++, i++) {
          int j = i | b;
          float
              eve_re = re[i],
              eve_im = im[i],
              odd_re = re[j],
              odd_im = im[j],
              w_re = PROUS[root * dr],
              w_im = PROUS[root * dr + 1],
              // odd *= w
              tmp_re = fma(odd_re, w_re, -odd_im * w_im);
          odd_im = fma(odd_re, w_im, odd_im * w_re);
          odd_re = tmp_re;
          re[i] = eve_re + odd_re;
          im[i] = eve_im + odd_im;
          re[j] = eve_re - odd_re;
          im[j] = eve_im - odd_im;
        }
      }
    }
  }

  public static void fftV1_64(double[] re, double[] im) {
    int n = re.length;
    if (n != im.length || n != len64() || !hasOneBitOrNone(n)) {
      throw new IllegalArgumentException();
    }

    for (int dr = n, b = 1; b < n; b <<= 1, dr >>= 1) {
      for (int i = 0; i < n; i += b) {
        for (int root = 0; root < b; root++, i++) {
          int j = i | b;
          double
              eve_re = re[i],
              eve_im = im[i],
              odd_re = re[j],
              odd_im = im[j],
              w_re = PROUS_64[root * dr],
              w_im = PROUS_64[root * dr + 1],
              // odd *= w
              tmp_re = fma(odd_re, w_re, -odd_im * w_im);
          odd_im = fma(odd_re, w_im, odd_im * w_re);
          odd_re = tmp_re;
          re[i] = eve_re + odd_re;
          im[i] = eve_im + odd_im;
          re[j] = eve_re - odd_re;
          im[j] = eve_im - odd_im;
        }
      }
    }
  }

  public static void fftV2(float[] real, float[] imag) {
    FloatVector
        im = FloatVector.fromArray(VSPEC_F32, imag, 0),
        re = FloatVector.fromArray(VSPEC_F32, real, 0);
    for (int lvl = 0; lvl < INDICES.length; ) {
      VectorShuffle<Float>
          eve = INDICES[lvl],
          odd = INDICES[lvl + 1];
      FloatVector
          eve_re = re.rearrange(eve),
          odd_re = re.rearrange(odd),
          eve_im = im.rearrange(eve),
          odd_im = im.rearrange(odd),
          w_re = FACTORS[lvl++],
          w_im = FACTORS[lvl++];
      im = odd_im.fma(w_re, odd_re.fma(w_im, eve_im));
      re = odd_im.fma(w_im.neg(), odd_re.fma(w_re, eve_re));
    }
    im.intoArray(imag, 0);
    re.intoArray(real, 0);
  }

  public static void fftV2_64(double[] real, double[] imag) {
    DoubleVector
        im = DoubleVector.fromArray(VSPEC_F64, imag, 0),
        re = DoubleVector.fromArray(VSPEC_F64, real, 0);
    for (int lvl = 0; lvl < INDICES_64.length; ) {
      VectorShuffle<Double>
          eve = INDICES_64[lvl],
          odd = INDICES_64[lvl + 1];
      DoubleVector
          eve_re = re.rearrange(eve),
          odd_re = re.rearrange(odd),
          eve_im = im.rearrange(eve),
          odd_im = im.rearrange(odd),
          w_re = FACTORS_64[lvl++],
          w_im = FACTORS_64[lvl++];
      im = odd_im.fma(w_re, odd_re.fma(w_im, eve_im));
      re = odd_im.fma(w_im.neg(), odd_re.fma(w_re, eve_re));
    }
    im.intoArray(imag, 0);
    re.intoArray(real, 0);
  }

  private void pass2(int product, PassData passData, double[][] twiddles) {
    final double[] data = passData.in;
    final double[] ret = passData.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData.inOffset;
    int j = passData.outOffset;
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

  private void pass2SIMD(int product, PassData passData, double[][] twiddles) {
    final double[] data = passData.in;
    final double[] ret = passData.out;
    final int factor = 2;
    final int m = n / factor;
    final int q = n / product;
    final int product_1 = product / factor;
    final int di = 2 * m;
    final int dj = 2 * product_1;
    int i = passData.inOffset;
    int j = passData.outOffset;
    final int dataInc = VSPEC_F64.length();
    final int k1inc = dataInc / 2;

    for (int k = 0; k < q; k++, j += dj) {
      final double[] twids = twiddles[k];
      DoubleVector w_r = DoubleVector.broadcast(VSPEC_F64, twids[0]);
      DoubleVector w_i = DoubleVector.broadcast(VSPEC_F64, -sign * twids[1]).mul(-1.0, mask);
      for (int k1 = 0; k1 < product_1; k1 += k1inc, i += dataInc, j += dataInc) {
        DoubleVector z0 = DoubleVector.fromArray(VSPEC_F64, data, i);
        DoubleVector z1 = DoubleVector.fromArray(VSPEC_F64, data, i + di);
        DoubleVector sum = z0.add(z1);
        sum.intoArray(ret, j);
        DoubleVector x = z0.sub(z1);
        DoubleVector sum2 = x.fma(w_r, x.mul(w_i).rearrange(shuffle));
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
  private record PassData(double[] in, int inOffset, double[] out, int outOffset) {
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

}
