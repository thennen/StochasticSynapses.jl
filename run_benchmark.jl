using Pkg
Pkg.activate("StochasticSynapses")
using StochasticSynapses
include(".\\benchmark\\benchmarks.jl")
#standard_benchmark_threaded("threads_order_benchmark")
#benchmark_read_write_threaded("threaded_IO_benchmark")
run_arrayofstruct_benchmarks()
