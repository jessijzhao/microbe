from timeit import default_timer
import torch
import torch.utils.benchmark as benchmark

from abc import abstractmethod


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BenchmarkType:
    PYTHON: str = "python"
    CUDA: str = "cuda"
    TORCH: str = "torch"


class BenchmarkFactory:

    class Benchmark:
        
        def run(self, **kwargs):

            # clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            
            runtime = self.get_runtime(**kwargs)
            
            # get memory
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated(device) * 1e-9

            return runtime, max_memory

        @abstractmethod
        def get_runtime(self, **kwargs):
            pass

    class CustomBenchmark(Benchmark):

        @abstractmethod
        def start_timer(self):
            pass

        @abstractmethod
        def end_timer(self):
            pass    

        def get_runtime(self, function, num_warmups, num_repeats):
            for _ in range(num_warmups):
                _ = function()
            
            torch.cuda.synchronize()

            self.start_timer()
            for _ in range(num_repeats):
                _ = function()

            return self.end_timer() /  num_repeats

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

        def get_runtime(self, function, num_warmups, num_repeats):
            if torch.cuda.is_available():
                return super().get_runtime(function, num_warmups, num_repeats)
            else:
                print("CUDA not available, skipping CUDA benchmark ...")

    class Torch(Benchmark):
        def get_runtime(self, function, num_repeats, **kwargs):
            # benchmark.Timer performs own warmups
            timer = benchmark.Timer(
                stmt="function()",
                globals={"function": function},
                num_threads=1
            )

            return timer.timeit(num_repeats).mean * 1000

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
