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

import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

import ffx.numerics.math.DoubleMath;
import ffx.numerics.multipole.CoulombTensorGlobal;
import ffx.numerics.multipole.CoulombTensorQI;
import ffx.numerics.multipole.GKEnergyGlobal;
import ffx.numerics.multipole.GKEnergyQI;
import ffx.numerics.multipole.PolarizableMultipole;
import ffx.numerics.multipole.QIFrame;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

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
public class NumericsBenchmark {

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
    double Es = 78.3;
    GKEnergyQI gkEnergyQI = new GKEnergyQI(gc, Es, true);
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

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
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
  @Fork(value = 1)
  public void coulombTensorGlobal(GlobalCoulombState state, Blackhole blackhole) {
    state.coulombTensorGlobal.generateTensor(r);
    double e = state.coulombTensorGlobal.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi,
        state.Fk, state.Ti, state.Tk);
    e += state.coulombTensorGlobal.polarizationEnergyAndGradient(state.mI, state.mK,
        1.0, 1.0, 1.0, state.Fi, state.Ti, state.Tk);
    blackhole.consume(e);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void gkTensorGlobal(GlobalGKState state, Blackhole blackhole) {
    double r2 = DoubleMath.length(r);
    state.gkEnergyGlobal.initPotential(r, r2, 2.0, 2.0);
    double e = state.gkEnergyGlobal.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi,
        state.Ti,
        state.Tk);
    e += state.gkEnergyGlobal.polarizationEnergyAndGradient(state.mI, state.mK, 1.0, state.Fi,
        state.Ti, state.Tk);
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
  @Fork(value = 1)
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
  @Fork(value = 1)
  public void gkTensorQI(QIGKState state, Blackhole blackhole) {
    state.qiFrame.setAndRotate(r, state.mI, state.mK);
    double r2 = DoubleMath.length(r);
    state.gkEnergyQI.initPotential(r, r2, 2.0, 2.0);
    double e = state.gkEnergyQI.multipoleEnergyAndGradient(state.mI, state.mK, state.Fi, state.Ti,
        state.Tk);
    e += state.gkEnergyQI.polarizationEnergyAndGradient(state.mI, state.mK, 1.0, state.Fi, state.Ti,
        state.Tk);
    state.gkEnergyQI.initBorn(r, r2, 2.0, 2.0);
    double db = state.gkEnergyQI.multipoleEnergyBornGrad(state.mI, state.mK);
    db += state.gkEnergyQI.polarizationEnergyBornGrad(state.mI, state.mK, true);
    state.qiFrame.toGlobal(state.Fi);
    state.qiFrame.toGlobal(state.Ti);
    state.qiFrame.toGlobal(state.Tk);
    blackhole.consume(e + db);
  }

}
