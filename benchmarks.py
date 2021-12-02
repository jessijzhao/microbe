from timeit import default_timer
import torch
import torch.utils.benchmark as benchmark

from abc import abstractmethod


class BenchmarkType:
    PYTHON: str = "python"
    CUDA: str = "cuda"
    TORCH: str = "torch"


class BenchmarkFactory:

    class Benchmark:

        @abstractmethod
        def run(self):
            pass

    class CustomBenchmark(Benchmark):

        @abstractmethod
        def start_timer(self):
            pass

        @abstractmethod
        def end_timer(self):
            pass    

        def run(self, function, num_warmups, num_runs):
            for _ in range(num_warmups):
                _ = function()

            torch.cuda.synchronize()

            self.start_timer()
            for _ in range(num_runs):
                _ = function()

            return self.end_timer() /  num_runs

    class Python(CustomBenchmark):
        def start_timer(self):
            self.start = default_timer()
        
        def end_timer(self):
            torch.cuda.synchronize()
            end = default_timer()
            return (end - self.start) * 1000

    class CUDA(CustomBenchmark):
        def start_timer(self):
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        
        def end_timer(self):
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)

        def run(self, function, num_warmups, num_runs):
            if torch.cuda.is_available():
                return super().run(function, num_warmups, num_runs)
            else:
                print("CUDA not available, skipping CUDA benchmark ...")

    class Torch(Benchmark):
        def run(self, function, num_runs, **kwargs):
            
            # benchmark.Timer performs own warmups
            timer = benchmark.Timer(
                stmt="function()",
                globals={"function": function},
                num_threads=1
            )

            return timer.timeit(num_runs).mean * 1000

    @staticmethod
    def create(benchmark_type: str):
        print(f"Creating {benchmark_type} benchmark ...")
        if benchmark_type == BenchmarkType.PYTHON:
            return BenchmarkFactory.Python()
        elif benchmark_type == BenchmarkType.CUDA:
            return BenchmarkFactory.CUDA()
        elif benchmark_type == BenchmarkType.TORCH:
            return BenchmarkFactory.Torch()
        else:
            print(f"Invalid benchnark type: {benchmark_type}.")
