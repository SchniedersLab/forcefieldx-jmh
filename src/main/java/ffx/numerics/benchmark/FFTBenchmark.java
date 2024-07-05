
package ffx.numerics.benchmark;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;
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

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scalarFFT(FFTState state, Blackhole blackhole) {
    fftV1(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scalarFFT_64(FFTState_64 state, Blackhole blackhole) {
    fftV1_64(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void vectorFFT(FFTState state, Blackhole blackhole) {
    fftV2(state.re, state.im);
    blackhole.consume(state.re);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void vectorFFT_64(FFTState_64 state, Blackhole blackhole) {
    fftV2_64(state.re, state.im);
    blackhole.consume(state.re);
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
}
