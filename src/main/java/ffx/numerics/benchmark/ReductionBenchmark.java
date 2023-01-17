package ffx.numerics.benchmark;

import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

import edu.rit.pj.ParallelTeam;
import edu.rit.pj.ParallelRegion;
import edu.rit.pj.IntegerForLoop;
import ffx.numerics.atomic.AtomicDoubleArray.AtomicDoubleArrayImpl;
import ffx.numerics.atomic.AtomicDoubleArray3D;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

import java.util.Random;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

public class ReductionBenchmark {

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

  private static final ParallelTeam parallelTeam = new ParallelTeam();
  private static final int nThreads = parallelTeam.getThreadCount();
  private static final int size = 10000;

  @State(Scope.Thread)
  public static class MultiState {

    public final AtomicDoubleArray3D multiArray3D =
        new AtomicDoubleArray3D(AtomicDoubleArrayImpl.MULTI, size, nThreads);
    public final AtomicDoubleArray3D pjArray3D =
        new AtomicDoubleArray3D(AtomicDoubleArrayImpl.PJ, size, nThreads);
    public final AtomicDoubleArray3D adderArray3D =
        new AtomicDoubleArray3D(AtomicDoubleArrayImpl.ADDER, size, nThreads);

    public void MultiState() {
      try {
        parallelTeam.execute(new ParallelRegion() {
          @Override
          public void run() throws Exception {
            execute(0, size - 1, new IntegerForLoop() {
              @Override
              public void run(int first, int last) {
                int threadID = getThreadIndex();
                Random random = new Random();
                for (int i = first; i <= last; i++) {
                  multiArray3D.set(threadID, i,
                      random.nextDouble(), random.nextDouble(), random.nextDouble());
                  pjArray3D.set(threadID, i,
                      random.nextDouble(), random.nextDouble(), random.nextDouble());
                  adderArray3D.set(threadID, i,
                      random.nextDouble(), random.nextDouble(), random.nextDouble());
                }
              }
            });
          }
        });
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void multiArrayReset(MultiState state, Blackhole blackhole) {
    state.multiArray3D.reset(parallelTeam, 0, size -1);
    blackhole.consume(state.multiArray3D);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void pjArrayReset(MultiState state, Blackhole blackhole) {
    state.pjArray3D.reset(parallelTeam, 0, size -1);
    blackhole.consume(state.pjArray3D);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void adderArrayReset(MultiState state, Blackhole blackhole) {
    state.adderArray3D.reset(parallelTeam, 0, size -1);
    blackhole.consume(state.adderArray3D);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void multiArrayReduce(MultiState state, Blackhole blackhole) {
    state.multiArray3D.reduce(parallelTeam, 0, size -1);
    blackhole.consume(state.multiArray3D);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void pjArrayReduce(MultiState state, Blackhole blackhole) {
    // state.pjArray3D.reduce(parallelTeam, 0, size -1);
    blackhole.consume(state.pjArray3D);
  }

  @Benchmark
  @BenchmarkMode(AverageTime)
  @OutputTimeUnit(NANOSECONDS)
  @Warmup(iterations = warmUpIterations, time = warmupTime)
  @Measurement(iterations = measurementIterations, time = measurementTime)
  @Fork(value = 1)
  public void adderArrayReduce(MultiState state, Blackhole blackhole) {
    // state.adderArray3D.reduce(parallelTeam, 0, size -1);
    blackhole.consume(state.adderArray3D);
  }

}
