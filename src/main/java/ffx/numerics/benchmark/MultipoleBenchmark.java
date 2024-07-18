/*
 * Copyright (c) 2014, Oracle America, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 *  * Neither the name of Oracle nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

package ffx.numerics.benchmark;

import ffx.numerics.math.DoubleMath;
import ffx.numerics.multipole.CoulombTensorGlobal;
import ffx.numerics.multipole.CoulombTensorGlobalSIMD;
import ffx.numerics.multipole.CoulombTensorQI;
import ffx.numerics.multipole.GKEnergyGlobal;
import ffx.numerics.multipole.GKEnergyQI;
import ffx.numerics.multipole.PolarizableMultipole;
import ffx.numerics.multipole.QIFrame;
import jdk.incubator.vector.DoubleVector;
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

import java.util.Arrays;

import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static jdk.incubator.vector.DoubleVector.fromArray;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

/**
 * JDK 11 Benchmark                              Mode  Cnt     Score    Error  Units
 * NumericsBenchmark.coulombTensorGlobal  avgt   25   279.105 ±  6.782  ns/op
 * NumericsBenchmark.coulombTensorQI      avgt   25   194.274 ±  4.617  ns/op
 * NumericsBenchmark.gkTensorGlobal       avgt   25  1291.491 ± 27.371  ns/op
 * NumericsBenchmark.gkTensorQI           avgt   25   943.652 ± 11.815  ns/op
 * NumericsBenchmark.rotateFrame          avgt   25    67.310 ±  0.751  ns/op
 * <p>
 * JDK 17 Benchmark                              Mode  Cnt     Score    Error  Units
 * NumericsBenchmark.coulombTensorGlobal  avgt   25   268.463 ±  3.011  ns/op
 * NumericsBenchmark.coulombTensorQI      avgt   25   196.708 ±  2.591  ns/op
 * NumericsBenchmark.gkTensorGlobal       avgt   25  1286.031 ± 14.329  ns/op
 * NumericsBenchmark.gkTensorQI           avgt   25   972.620 ± 11.490  ns/op
 * NumericsBenchmark.rotateFrame          avgt   25    69.586 ±  1.199  ns/op
 * <p>
 * JDK 19 Benchmark                              Mode  Cnt     Score    Error  Units
 * NumericsBenchmark.coulombTensorGlobal  avgt   25   270.668 ±  5.640  ns/op
 * NumericsBenchmark.coulombTensorQI      avgt   25   206.313 ±  6.197  ns/op
 * NumericsBenchmark.gkTensorGlobal       avgt   25  1334.656 ± 32.395  ns/op
 * NumericsBenchmark.gkTensorQI           avgt   25  1010.790 ± 28.696  ns/op
 * NumericsBenchmark.rotateFrame          avgt   25    69.300 ±  1.768  ns/op
 */
public class MultipoleBenchmark {

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

  // Water Dimer O-O interaction
  protected static final double[] r = {2.97338529, 0.0, 0.03546452};
  // Rotated multipole at site I.
  protected final static double[] Qi = {-0.51966,
      0.06979198988239577, 0.0, 0.0289581620819011,
      0.024871041044109393, -0.1170771231287098, 0.09220608208460039,
      0.0, -0.03374891685535346, 0.0};
  // Rotated multipole at site K.
  protected final static double[] Qk = {-0.51966,
      0.05872406108747119, 0.0, 0.047549780780788455,
      0.048623413357888695, -0.1170771231287098, 0.06845370977082109,
      0.0, -0.04662811558421081, 0.0};

  protected final static double[] Ui = {0.04886563833303603, 0.0, -0.0018979726219775425};
  protected final static double[] Uk = {-0.040839567654139396, 0.0, -5.982126263609587E-4};

  protected final static VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

  protected final static int vectorLength = DoubleVector.zero(SPECIES).length();
  protected final static int multipoleSize = 10;
  protected final static double[] VQi = new double[multipoleSize * vectorLength];
  protected final static double[] VQk = new double[multipoleSize * vectorLength];
  protected final static double[] VUi = new double[3 * vectorLength];
  protected final static double[] VUk = new double[3 * vectorLength];
  protected final static double[] xyz = new double[3 * vectorLength];

  // Order 0
  protected final static int T000 = 0;
  // l + m + n = 1 (3)   4
  protected final static int T100 = vectorLength;
  protected final static int T010 = 2 * vectorLength;
  protected final static int T001 = 3 * vectorLength;
  // l + m + n = 2 (6)  10
  protected final static int T200 = 4 * vectorLength;
  protected final static int T020 = 5 * vectorLength;
  protected final static int T002 = 6 * vectorLength;
  protected final static int T110 = 7 * vectorLength;
  protected final static int T101 = 8 * vectorLength;
  protected final static int T011 = 9 * vectorLength;
  // l + m + n = 3 (10) 20
  protected final static int T300 = 10 * vectorLength;
  protected final static int T030 = 11 * vectorLength;
  protected final static int T003 = 12 * vectorLength;
  protected final static int T210 = 13 * vectorLength;
  protected final static int T201 = 14 * vectorLength;
  protected final static int T120 = 15 * vectorLength;
  protected final static int T021 = 16 * vectorLength;
  protected final static int T102 = 17 * vectorLength;
  protected final static int T012 = 18 * vectorLength;
  protected final static int T111 = 19 * vectorLength;
  // l + m + n = 4 (15) 35
  protected final static int T400 = 20 * vectorLength;
  protected final static int T040 = 21 * vectorLength;
  protected final static int T004 = 22 * vectorLength;
  protected final static int T310 = 23 * vectorLength;
  protected final static int T301 = 24 * vectorLength;
  protected final static int T130 = 25 * vectorLength;
  protected final static int T031 = 26 * vectorLength;
  protected final static int T103 = 27 * vectorLength;
  protected final static int T013 = 28 * vectorLength;
  protected final static int T220 = 29 * vectorLength;
  protected final static int T202 = 30 * vectorLength;
  protected final static int T022 = 31 * vectorLength;
  protected final static int T211 = 32 * vectorLength;
  protected final static int T121 = 33 * vectorLength;
  protected final static int T112 = 34 * vectorLength;
  // l + m + n = 5 (21) 56
  protected final static int T500 = 35 * vectorLength;
  protected final static int T050 = 36 * vectorLength;
  protected final static int T005 = 37 * vectorLength;
  protected final static int T410 = 38 * vectorLength;
  protected final static int T401 = 39 * vectorLength;
  protected final static int T140 = 40 * vectorLength;
  protected final static int T041 = 41 * vectorLength;
  protected final static int T104 = 42 * vectorLength;
  protected final static int T014 = 43 * vectorLength;
  protected final static int T320 = 44 * vectorLength;
  protected final static int T302 = 45 * vectorLength;
  protected final static int T230 = 46 * vectorLength;
  protected final static int T032 = 47 * vectorLength;
  protected final static int T203 = 48 * vectorLength;
  protected final static int T023 = 49 * vectorLength;
  protected final static int T311 = 50 * vectorLength;
  protected final static int T131 = 51 * vectorLength;
  protected final static int T113 = 52 * vectorLength;
  protected final static int T221 = 53 * vectorLength;
  protected final static int T212 = 54 * vectorLength;
  protected final static int T122 = 55 * vectorLength;

  static {
    int index = 0;
    for (int i = 0; i < 10; i++) {
      for (int v = 0; v < vectorLength; v++) {
        VQi[index] = Qi[i];
        VQk[index++] = Qk[i];
      }
    }

    index = 0;
    for (int i = 0; i < 3; i++) {
      for (int v = 0; v < vectorLength; v++) {
        VUi[index] = Ui[i];
        VUk[index] = Uk[i];
        xyz[index] = r[i];
        index++;
      }
    }
  }

  @State(Scope.Thread)
  public static class GlobalCoulombState {

    double[] Fi = new double[3];
    double[] Fk = new double[3];
    double[] Ti = new double[3];
    double[] Tk = new double[3];
    int order = 5;
    CoulombTensorGlobal coulombTensorGlobal = new CoulombTensorGlobal(order);
    PolarizableMultipole mI = new PolarizableMultipole(Qi, Ui, Ui);
    PolarizableMultipole mK = new PolarizableMultipole(Qk, Uk, Uk);
  }

  @State(Scope.Thread)
  public static class QICoulombState {

    double[] Fi = new double[3];
    double[] Fk = new double[3];
    double[] Ti = new double[3];
    double[] Tk = new double[3];
    int order = 5;
    CoulombTensorQI coulombTensorQI = new CoulombTensorQI(order);
    PolarizableMultipole mI = new PolarizableMultipole(Qi, Ui, Ui);
    PolarizableMultipole mK = new PolarizableMultipole(Qk, Uk, Uk);
    QIFrame qiFrame = new QIFrame();
  }

  @State(Scope.Thread)
  public static class GlobalGKState {

    double[] Fi = new double[3];
    double[] Ti = new double[3];
    double[] Tk = new double[3];
    double gc = 2.455;
    double Es = 78.3;
    GKEnergyGlobal gkEnergyGlobal = new GKEnergyGlobal(gc, Es, true);
    PolarizableMultipole mI = new PolarizableMultipole(Qi, Ui, Ui);
    PolarizableMultipole mK = new PolarizableMultipole(Qk, Uk, Uk);
  }

  @State(Scope.Thread)
  public static class QIGKState {

    double[] Fi = new double[3];
    double[] Ti = new double[3];
    double[] Tk = new double[3];
    double gc = 2.455;
    double Eh = 1.0;
    double Es = 78.3;
    GKEnergyQI gkEnergyQI = new GKEnergyQI(Eh, Es, gc, true);
    PolarizableMultipole mI = new PolarizableMultipole(Qi, Ui, Ui);
    PolarizableMultipole mK = new PolarizableMultipole(Qk, Uk, Uk);
    QIFrame qiFrame = new QIFrame();
  }

  @State(Scope.Thread)
  public static class QIState {

    PolarizableMultipole mI = new PolarizableMultipole(Qi, Ui, Ui);
    PolarizableMultipole mK = new PolarizableMultipole(Qk, Uk, Uk);
    double[] Fi = new double[3];
    double[] Ti = new double[3];
    double[] Tk = new double[3];
    QIFrame qiFrame = new QIFrame();
  }

  @State(Scope.Thread)
  public static class CoulombGlobalStateSIMD {
    int order = 5;
    double[] work;
    double[] r = xyz;
    double[] tensor = new double[vectorLength * 56];
    double[] e = new double[vectorLength * 20];
    double[] Qi = Arrays.copyOf(VQi, VQi.length);
    CoulombTensorGlobalSIMD coulombTensorGlobal = new CoulombTensorGlobalSIMD(order);

    public CoulombGlobalStateSIMD() {
      work = new double[vectorLength * order];
    }
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void rotateFrame(QIState state) {
    state.qiFrame.setAndRotate(r, state.mI, state.mK);
    state.qiFrame.toGlobal(state.Fi);
    state.qiFrame.toGlobal(state.Ti);
    state.qiFrame.toGlobal(state.Tk);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void coulombTensorGlobal(GlobalCoulombState state, Blackhole blackhole) {
    state.coulombTensorGlobal.generateTensor(r);
    double e = state.coulombTensorGlobal.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi, state.Fk, state.Ti, state.Tk);
    e += state.coulombTensorGlobal.polarizationEnergyAndGradient(state.mI, state.mK,
        1.0, 1.0, 1.0, state.Fi, state.Ti, state.Tk);
    blackhole.consume(e);
  }


  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {
      "--add-modules=jdk.incubator.vector",
  })
  public void contractTensorGlobalSIMDFMA(CoulombGlobalStateSIMD state, Blackhole blackhole) {
    double[] e = state.e;
    double[] t = state.tensor;
    double[] Qi = state.Qi;
    DoubleVector q = fromArray(SPECIES, Qi, T000);
    DoubleVector dx = fromArray(SPECIES, Qi, T100);
    DoubleVector dy = fromArray(SPECIES, Qi, T010);
    DoubleVector dz = fromArray(SPECIES, Qi, T001);
    DoubleVector qxx = fromArray(SPECIES, Qi, T200);
    DoubleVector qyy = fromArray(SPECIES, Qi, T020);
    DoubleVector qzz = fromArray(SPECIES, Qi, T002);
    DoubleVector qxy = fromArray(SPECIES, Qi, T110);
    DoubleVector qxz = fromArray(SPECIES, Qi, T101);
    DoubleVector qyz = fromArray(SPECIES, Qi, T011);
    // Order 0
    DoubleVector t000 = fromArray(SPECIES, t, T000);
    DoubleVector term000 = q.mul(t000);
    DoubleVector t100 = fromArray(SPECIES, t, T100);
    term000 = dx.fma(t100, term000);
    DoubleVector t010 = fromArray(SPECIES, t, T010);
    term000 = dy.fma(t010, term000);
    DoubleVector t001 = fromArray(SPECIES, t, T001);
    term000 = dz.fma(t001, term000);
    DoubleVector t200 = fromArray(SPECIES, t, T200);
    term000 = qxx.fma(t200, term000);
    DoubleVector t020 = fromArray(SPECIES, t, T020);
    term000 = qyy.fma(t020, term000);
    DoubleVector t002 = fromArray(SPECIES, t, T002);
    term000 = qzz.fma(t002, term000);
    DoubleVector t110 = fromArray(SPECIES, t, T110);
    term000 = qxy.fma(t110, term000);
    DoubleVector t101 = fromArray(SPECIES, t, T101);
    term000 = qxz.fma(t101, term000);
    DoubleVector t011 = fromArray(SPECIES, t, T011);
    term000 = qyz.fma(t011, term000);
    term000.intoArray(e, T000);
    // Order 1
    // l + m + n = 1 (3)   4
    DoubleVector term100 = q.mul(t100);
    term100 = dx.fma(t200, term100);
    term100 = dy.fma(t110, term100);
    term100 = dz.fma(t101, term100);
    DoubleVector t300 = fromArray(SPECIES, t, T300);
    term100 = qxx.fma(t300, term100);
    DoubleVector t120 = fromArray(SPECIES, t, T120);
    term100 = qyy.fma(t120, term100);
    DoubleVector t102 = fromArray(SPECIES, t, T102);
    term100 = qzz.fma(t102, term100);
    DoubleVector t210 = fromArray(SPECIES, t, T210);
    term100 = qxy.fma(t210, term100);
    DoubleVector t201 = fromArray(SPECIES, t, T201);
    term100 = qxz.fma(t201, term100);
    DoubleVector t111 = fromArray(SPECIES, t, T111);
    term100 = qyz.fma(t111, term100);
    term100.neg().intoArray(e, T100);
    DoubleVector term010 = q.mul(t010);
    term010 = dx.fma(t110, term010);
    term010 = dy.fma(t020, term010);
    term010 = dz.fma(t011, term010);
    term010 = qxx.fma(t210, term010);
    DoubleVector t030 = fromArray(SPECIES, t, T030);
    term010 = qyy.fma(t030, term010);
    DoubleVector t012 = fromArray(SPECIES, t, T012);
    term010 = qzz.fma(t012, term010);
    term010 = qxy.fma(t120, term010);
    term010 = qxz.fma(t111, term010);
    DoubleVector t021 = fromArray(SPECIES, t, T021);
    term010 = qyz.fma(t021, term010);
    term010.neg().intoArray(e, T010);
    DoubleVector term001 = q.mul(t001);
    term001 = dx.fma(t101, term001);
    term001 = dy.fma(t011, term001);
    term001 = dz.fma(t002, term001);
    term001 = qxx.fma(t201, term001);
    term001 = qyy.fma(t021, term001);
    DoubleVector t003 = fromArray(SPECIES, t, T003);
    term001 = qzz.fma(t003, term001);
    term001 = qxy.fma(t111, term001);
    term001 = qxz.fma(t102, term001);
    term001 = qyz.fma(t012, term001);
    term001.neg().intoArray(e, T001);
    // Order 2
    // l + m + n = 2 (6)  10
    DoubleVector term200 = q.mul(t200);
    term200 = dx.fma(t300, term200);
    term200 = dy.fma(t210, term200);
    term200 = dz.fma(t201, term200);
    DoubleVector t400 = fromArray(SPECIES, t, T400);
    term200 = qxx.fma(t400, term200);
    DoubleVector t220 = fromArray(SPECIES, t, T220);
    term200 = qyy.fma(t220, term200);
    DoubleVector t202 = fromArray(SPECIES, t, T202);
    term200 = qzz.fma(t202, term200);
    DoubleVector t310 = fromArray(SPECIES, t, T310);
    term200 = qxy.fma(t310, term200);
    DoubleVector t301 = fromArray(SPECIES, t, T301);
    term200 = qxz.fma(t301, term200);
    DoubleVector t211 = fromArray(SPECIES, t, T211);
    term200 = qyz.fma(t211, term200);
    term200.intoArray(e, T200);
    DoubleVector term020 = q.mul(t020);
    term020 = dx.fma(t120, term020);
    term020 = dy.fma(t030, term020);
    term020 = dz.fma(t021, term020);
    term020 = qxx.fma(t220, term020);
    DoubleVector t040 = fromArray(SPECIES, t, T040);
    term020 = qyy.fma(t040, term020);
    DoubleVector t022 = fromArray(SPECIES, t, T022);
    term020 = qzz.fma(t022, term020);
    DoubleVector t130 = fromArray(SPECIES, t, T130);
    term020 = qxy.fma(t130, term020);
    DoubleVector t121 = fromArray(SPECIES, t, T121);
    term020 = qxz.fma(t121, term020);
    DoubleVector t031 = fromArray(SPECIES, t, T031);
    term020 = qyz.fma(t031, term020);
    term020.intoArray(e, T020);
    DoubleVector term002 = q.mul(t002);
    term002 = dx.fma(t102, term002);
    term002 = dy.fma(t012, term002);
    term002 = dz.fma(t003, term002);
    term002 = qxx.fma(t202, term002);
    term002 = qyy.fma(t022, term002);
    DoubleVector t004 = fromArray(SPECIES, t, T004);
    term002 = qzz.fma(t004, term002);
    DoubleVector t112 = fromArray(SPECIES, t, T112);
    term002 = qxy.fma(t112, term002);
    DoubleVector t103 = fromArray(SPECIES, t, T103);
    term002 = qxz.fma(t103, term002);
    DoubleVector t013 = fromArray(SPECIES, t, T013);
    term002 = qyz.fma(t013, term002);
    term002.intoArray(e, T002);
    DoubleVector term110 = q.mul(t110);
    term110 = dx.fma(t210, term110);
    term110 = dy.fma(t120, term110);
    term110 = dz.fma(t111, term110);
    term110 = qxx.fma(t310, term110);
    term110 = qyy.fma(t130, term110);
    term110 = qzz.fma(t112, term110);
    term110 = qxy.fma(t220, term110);
    term110 = qxz.fma(t211, term110);
    term110 = qyz.fma(t121, term110);
    term110.intoArray(e, T110);
    DoubleVector term101 = q.mul(t101);
    term101 = dx.fma(t201, term101);
    term101 = dy.fma(t111, term101);
    term101 = dz.fma(t102, term101);
    term101 = qxx.fma(t301, term101);
    term101 = qyy.fma(t121, term101);
    term101 = qzz.fma(t103, term101);
    term101 = qxy.fma(t211, term101);
    term101 = qxz.fma(t202, term101);
    term101 = qyz.fma(t112, term101);
    term101.intoArray(e, T101);
    DoubleVector term011 = q.mul(t011);
    term011 = dx.fma(t111, term011);
    term011 = dy.fma(t021, term011);
    term011 = dz.fma(t012, term011);
    term011 = qxx.fma(t211, term011);
    term011 = qyy.fma(t031, term011);
    term011 = qzz.fma(t013, term011);
    term011 = qxy.fma(t121, term011);
    term011 = qxz.fma(t112, term011);
    term011 = qyz.fma(t022, term011);
    term011.intoArray(e, T011);
    // Order 3
    // l + m + n = 3 (10) 20
    DoubleVector term300 = q.mul(t300);
    term300 = dx.fma(t400, term300);
    term300 = dy.fma(t310, term300);
    term300 = dz.fma(t301, term300);
    DoubleVector t500 = fromArray(SPECIES, t, T500);
    term300 = qxx.fma(t500, term300);
    DoubleVector t320 = fromArray(SPECIES, t, T320);
    term300 = qyy.fma(t320, term300);
    DoubleVector t302 = fromArray(SPECIES, t, T302);
    term300 = qzz.fma(t302, term300);
    DoubleVector t410 = fromArray(SPECIES, t, T410);
    term300 = qxy.fma(t410, term300);
    DoubleVector t401 = fromArray(SPECIES, t, T401);
    term300 = qxz.fma(t401, term300);
    DoubleVector t311 = fromArray(SPECIES, t, T311);
    term300 = qyz.fma(t311, term300);
    term300.neg().intoArray(e, T300);
    DoubleVector term030 = q.mul(t030);
    term030 = dx.fma(t130, term030);
    term030 = dy.fma(t040, term030);
    term030 = dz.fma(t031, term030);
    DoubleVector t230 = fromArray(SPECIES, t, T230);
    term030 = qxx.fma(t230, term030);
    DoubleVector t050 = fromArray(SPECIES, t, T050);
    term030 = qyy.fma(t050, term030);
    DoubleVector t032 = fromArray(SPECIES, t, T032);
    term030 = qzz.fma(t032, term030);
    DoubleVector t140 = fromArray(SPECIES, t, T140);
    term030 = qxy.fma(t140, term030);
    DoubleVector t131 = fromArray(SPECIES, t, T131);
    term030 = qxz.fma(t131, term030);
    DoubleVector t041 = fromArray(SPECIES, t, T041);
    term030 = qyz.fma(t041, term030);
    term030.neg().intoArray(e, T030);
    DoubleVector term003 = q.mul(t003);
    term003 = dx.fma(t103, term003);
    term003 = dy.fma(t013, term003);
    term003 = dz.fma(t004, term003);
    DoubleVector t203 = fromArray(SPECIES, t, T203);
    term003 = qxx.fma(t203, term003);
    DoubleVector t023 = fromArray(SPECIES, t, T023);
    term003 = qyy.fma(t023, term003);
    DoubleVector t005 = fromArray(SPECIES, t, T005);
    term003 = qzz.fma(t005, term003);
    DoubleVector t113 = fromArray(SPECIES, t, T113);
    term003 = qxy.fma(t113, term003);
    DoubleVector t104 = fromArray(SPECIES, t, T104);
    term003 = qxz.fma(t104, term003);
    DoubleVector t014 = fromArray(SPECIES, t, T014);
    term003 = qyz.fma(t014, term003);
    term003.neg().intoArray(e, T003);
    DoubleVector term210 = q.mul(t210);
    term210 = dx.fma(t310, term210);
    term210 = dy.fma(t220, term210);
    term210 = dz.fma(t211, term210);
    term210 = qxx.fma(t410, term210);
    term210 = qyy.fma(t230, term210);
    DoubleVector t212 = fromArray(SPECIES, t, T212);
    term210 = qzz.fma(t212, term210);
    term210 = qxy.fma(t320, term210);
    term210 = qxz.fma(t311, term210);
    DoubleVector t221 = fromArray(SPECIES, t, T221);
    term210 = qyz.fma(t221, term210);
    term210.neg().intoArray(e, T210);
    DoubleVector term201 = q.mul(t201);
    term201 = dx.fma(t301, term201);
    term201 = dy.fma(t211, term201);
    term201 = dz.fma(t202, term201);
    term201 = qxx.fma(t401, term201);
    term201 = qyy.fma(t221, term201);
    term201 = qzz.fma(t203, term201);
    term201 = qxy.fma(t311, term201);
    term201 = qxz.fma(t302, term201);
    term201 = qyz.fma(t212, term201);
    term201.neg().intoArray(e, T201);
    DoubleVector term120 = q.mul(t120);
    term120 = dx.fma(t220, term120);
    term120 = dy.fma(t130, term120);
    term120 = dz.fma(t121, term120);
    term120 = qxx.fma(t320, term120);
    term120 = qyy.fma(t140, term120);
    DoubleVector t122 = fromArray(SPECIES, t, T122);
    term120 = qzz.fma(t122, term120);
    term120 = qxy.fma(t230, term120);
    term120 = qxz.fma(t221, term120);
    term120 = qyz.fma(t131, term120);
    term120.neg().intoArray(e, T120);
    DoubleVector term021 = q.mul(t021);
    term021 = dx.fma(t121, term021);
    term021 = dy.fma(t031, term021);
    term021 = dz.fma(t022, term021);
    term021 = qxx.fma(t221, term021);
    term021 = qyy.fma(t041, term021);
    term021 = qzz.fma(t023, term021);
    term021 = qxy.fma(t131, term021);
    term021 = qxz.fma(t122, term021);
    term021 = qyz.fma(t032, term021);
    term021.neg().intoArray(e, T021);
    DoubleVector term102 = q.mul(t102);
    term102 = dx.fma(t202, term102);
    term102 = dy.fma(t112, term102);
    term102 = dz.fma(t103, term102);
    term102 = qxx.fma(t302, term102);
    term102 = qyy.fma(t122, term102);
    term102 = qzz.fma(t104, term102);
    term102 = qxy.fma(t212, term102);
    term102 = qxz.fma(t203, term102);
    term102 = qyz.fma(t113, term102);
    term102.neg().intoArray(e, T102);
    DoubleVector term012 = q.mul(t012);
    term012 = dx.fma(t112, term012);
    term012 = dy.fma(t022, term012);
    term012 = dz.fma(t013, term012);
    term012 = qxx.fma(t212, term012);
    term012 = qyy.fma(t032, term012);
    term012 = qzz.fma(t014, term012);
    term012 = qxy.fma(t122, term012);
    term012 = qxz.fma(t113, term012);
    term012 = qyz.fma(t023, term012);
    term012.neg().intoArray(e, T012);
    DoubleVector term111 = q.mul(t111);
    term111 = dx.fma(t211, term111);
    term111 = dy.fma(t121, term111);
    term111 = dz.fma(t112, term111);
    term111 = qxx.fma(t311, term111);
    term111 = qyy.fma(t131, term111);
    term111 = qzz.fma(t113, term111);
    term111 = qxy.fma(t221, term111);
    term111 = qxz.fma(t212, term111);
    term111 = qyz.fma(t122, term111);
    term111.neg().intoArray(e, T111);
    blackhole.consume(e);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {
      "--add-modules=jdk.incubator.vector",
  })
  public void coulombTensorGlobalSIMDFMA(CoulombGlobalStateSIMD state, Blackhole blackhole) {
    double[] work = state.work;
    double[] r = state.r;
    double[] t = state.tensor;
    DoubleVector term0000 = fromArray(SPECIES, work, 0);
    DoubleVector x = fromArray(SPECIES, r, 0);
    DoubleVector y = fromArray(SPECIES, r, vectorLength);
    DoubleVector z = fromArray(SPECIES, r, vectorLength * 2);
    term0000.intoArray(t, T000);
    // l + m + n = 1 (3)   4
    DoubleVector term0001 = fromArray(SPECIES, work, vectorLength);
    x.mul(term0001).intoArray(t, T100);
    y.mul(term0001).intoArray(t, T010);
    z.mul(term0001).intoArray(t, T001);
    // l + m + n = 2 (6)  10
    DoubleVector term0002 = fromArray(SPECIES, work, vectorLength * 2);
    DoubleVector term1001 = x.mul(term0002);
    x.fma(term1001, term0001).intoArray(t, T200);
    DoubleVector term0101 = y.mul(term0002);
    y.fma(term0101, term0001).intoArray(t, T020);
    DoubleVector term0011 = z.mul(term0002);
    z.fma(term0011, term0001).intoArray(t, T002);
    y.mul(term1001).intoArray(t, T110);
    z.mul(term1001).intoArray(t, T101);
    z.mul(term0101).intoArray(t, T011);
    // l + m + n = 3 (10) 20
    DoubleVector term0003 = fromArray(SPECIES, work, vectorLength * 3);
    DoubleVector term1002 = x.mul(term0003);
    DoubleVector term2001 = x.fma(term1002, term0002);
    x.fma(term2001, term1001.mul(2)).intoArray(t, T300);
    DoubleVector term0102 = y.mul(term0003);
    DoubleVector term0201 = y.fma(term0102, term0002);
    y.fma(term0201, term0101.mul(2)).intoArray(t, T030);
    DoubleVector term0012 = z.mul(term0003);
    DoubleVector term0021 = z.fma(term0012, term0002);
    z.fma(term0021, term0011.mul(2)).intoArray(t, T003);
    y.mul(term2001).intoArray(t, T210);
    z.mul(term2001).intoArray(t, T201);
    DoubleVector term1101 = y.mul(term1002);
    y.fma(term1101, term1001).intoArray(t, T120);
    z.mul(term0201).intoArray(t, T021);
    DoubleVector term1011 = z.mul(term1002);
    z.fma(term1011, term1001).intoArray(t, T102);
    DoubleVector term0111 = z.mul(term0102);
    z.fma(term0111, term0101).intoArray(t, T012);
    z.mul(term1101).intoArray(t, T111);
    // l + m + n = 4 (15) 35
    DoubleVector term0004 = fromArray(SPECIES, work, vectorLength * 4);
    DoubleVector term1003 = x.mul(term0004);
    DoubleVector term2002 = x.fma(term1003, term0003);
    DoubleVector term3001 = x.fma(term2002, term1002.mul(2));
    x.fma(term3001, term2001.mul(3)).intoArray(t, T400);
    DoubleVector term0103 = y.mul(term0004);
    DoubleVector term0202 = y.fma(term0103, term0003);
    DoubleVector term0301 = y.fma(term0202, term0102.mul(2));
    y.fma(term0301, term0201.mul(3)).intoArray(t, T040);
    DoubleVector term0013 = z.mul(term0004);
    DoubleVector term0022 = z.fma(term0013, term0003);
    DoubleVector term0031 = z.fma(term0022, term0012.mul(2));
    z.fma(term0031, term0021.mul(3)).intoArray(t, T004);
    y.mul(term3001).intoArray(t, T310);
    z.mul(term3001).intoArray(t, T301);
    DoubleVector term1102 = y.mul(term1003);
    DoubleVector term1201 = y.fma(term1102, term1002);
    y.fma(term1201, term1101.mul(2)).intoArray(t, T130);
    z.mul(term0301).intoArray(t, T031);
    DoubleVector term1012 = z.mul(term1003);
    DoubleVector term1021 = z.fma(term1012, term1002);
    z.fma(term1021, term1011.mul(2)).intoArray(t, T103);
    DoubleVector term0112 = z.mul(term0103);
    DoubleVector term0121 = z.fma(term0112, term0102);
    z.fma(term0121, term0111.mul(2)).intoArray(t, T013);
    DoubleVector term2101 = y.mul(term2002);
    y.fma(term2101, term2001).intoArray(t, T220);
    DoubleVector term2011 = z.mul(term2002);
    z.fma(term2011, term2001).intoArray(t, T202);
    DoubleVector term0211 = z.mul(term0202);
    z.fma(term0211, term0201).intoArray(t, T022);
    z.mul(term2101).intoArray(t, T211);
    z.mul(term1201).intoArray(t, T121);
    DoubleVector term1111 = z.mul(term1102);
    z.fma(term1111, term1101).intoArray(t, T112);
    // l + m + n = 5 (21) 56
    DoubleVector term0005 = fromArray(SPECIES, work, vectorLength * 4);
    DoubleVector term1004 = x.mul(term0005);
    DoubleVector term2003 = x.fma(term1004, term0004);
    DoubleVector term3002 = x.fma(term2003, term1003.mul(2));
    DoubleVector term4001 = x.fma(term3002, term2002.mul(3));
    x.fma(term4001, term3001.mul(4)).intoArray(t, T500);
    DoubleVector term0104 = y.mul(term0005);
    DoubleVector term0203 = y.fma(term0104, term0004);
    DoubleVector term0302 = y.fma(term0203, term0103.mul(2));
    DoubleVector term0401 = y.fma(term0302, term0202.mul(3));
    y.fma(term0401, term0301.mul(4)).intoArray(t, T050);
    DoubleVector term0014 = z.mul(term0005);
    DoubleVector term0023 = z.fma(term0014, term0004);
    DoubleVector term0032 = z.fma(term0023, term0013.mul(2));
    DoubleVector term0041 = z.fma(term0032, term0022.mul(3));
    z.fma(term0041, term0031.mul(4)).intoArray(t, T005);
    y.mul(term4001).intoArray(t, T410);
    DoubleVector term1103 = y.mul(term1004);
    DoubleVector term1202 = y.fma(term1103, term1003);
    DoubleVector term1301 = y.fma(term1202, term1102.mul(2));
    z.mul(term4001).intoArray(t, T401);
    y.fma(term1301, term1201.mul(3)).intoArray(t, T140);
    DoubleVector term1013 = z.mul(term1004);
    DoubleVector term1022 = z.fma(term1013, term1003);
    DoubleVector term1031 = z.fma(term1022, term1012.mul(2));
    z.mul(term0401).intoArray(t, T041);
    z.fma(term1031, term1021.mul(3)).intoArray(t, T104);
    DoubleVector term0113 = z.mul(term0104);
    DoubleVector term0122 = z.fma(term0113, term0103);
    DoubleVector term0131 = z.fma(term0122, term0112.mul(2));
    z.fma(term0131, term0121.mul(3)).intoArray(t, T014);
    DoubleVector term3101 = y.mul(term3002);
    y.fma(term3101, term3001).intoArray(t, T320);
    DoubleVector term3011 = z.mul(term3002);
    z.fma(term3011, term3001).intoArray(t, T302);
    DoubleVector term2102 = y.mul(term2003);
    DoubleVector term2201 = y.fma(term2102, term2002);
    y.fma(term2201, term2101.mul(2)).intoArray(t, T230);
    DoubleVector term0311 = z.mul(term0302);
    z.fma(term0311, term0301).intoArray(t, T032);
    DoubleVector term2012 = z.mul(term2003);
    DoubleVector term2021 = z.fma(term2012, term2002);
    z.fma(term2021, term2011.mul(2)).intoArray(t, T203);
    DoubleVector term0212 = z.mul(term0203);
    DoubleVector term0221 = z.fma(term0212, term0202);
    z.fma(term0221, term0211.mul(2)).intoArray(t, T023);
    z.mul(term3101).intoArray(t, T311);
    z.mul(term1301).intoArray(t, T131);
    DoubleVector term1112 = z.mul(term1103);
    DoubleVector term1121 = z.fma(term1112, term1102);
    z.fma(term1121, term1111.mul(2)).intoArray(t, T113);
    z.mul(term2201).intoArray(t, T221);
    DoubleVector term2111 = z.mul(term2102);
    z.fma(term2111, term2101).intoArray(t, T212);
    DoubleVector term1211 = z.mul(term1202);
    z.fma(term1211, term1201).intoArray(t, T122);
    blackhole.consume(t);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void gkTensorGlobal(GlobalGKState state, Blackhole blackhole) {
    double r2 = DoubleMath.length(r);
    state.gkEnergyGlobal.initPotential(r, r2, 2.0, 2.0);
    double e = state.gkEnergyGlobal.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi, state.Ti, state.Tk);
    e += state.gkEnergyGlobal.polarizationEnergyAndGradient(state.mI, state.mK, 1.0, state.Fi, state.Ti, state.Tk);
    state.gkEnergyGlobal.initBorn(r, r2, 2.0, 2.0);
    double db = state.gkEnergyGlobal.multipoleEnergyBornGrad(state.mI, state.mK);
    db += state.gkEnergyGlobal.polarizationEnergyBornGrad(state.mI, state.mK, true);
    blackhole.consume(e + db);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void coulombTensorQI(QICoulombState state, Blackhole blackhole) {
    state.qiFrame.setAndRotate(r, state.mI, state.mK);
    state.coulombTensorQI.generateTensor(r);
    double e = state.coulombTensorQI.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi,
        state.Fk, state.Ti, state.Tk);
    e += state.coulombTensorQI.polarizationEnergyAndGradient(state.mI, state.mK,
        1.0, 1.0, 1.0, state.Fi, state.Ti, state.Tk);
    state.qiFrame.toGlobal(state.Fi);
    state.qiFrame.toGlobal(state.Ti);
    state.qiFrame.toGlobal(state.Tk);
    blackhole.consume(e);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void gkTensorQI(QIGKState state, Blackhole blackhole) {
    state.qiFrame.setAndRotate(r, state.mI, state.mK);
    double r2 = DoubleMath.length(r);
    state.gkEnergyQI.initPotential(r, r2, 2.0, 2.0);
    double e = state.gkEnergyQI.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi, state.Ti, state.Tk);
    e += state.gkEnergyQI.polarizationEnergyAndGradient(state.mI, state.mK, 1.0, state.Fi, state.Ti, state.Tk);
    state.gkEnergyQI.initBorn(r, r2, 2.0, 2.0);
    double db = state.gkEnergyQI.multipoleEnergyBornGrad(state.mI, state.mK);
    db += state.gkEnergyQI.polarizationEnergyBornGrad(state.mI, state.mK, true);
    state.qiFrame.toGlobal(state.Fi);
    state.qiFrame.toGlobal(state.Ti);
    state.qiFrame.toGlobal(state.Tk);
    blackhole.consume(e + db);
  }

}
