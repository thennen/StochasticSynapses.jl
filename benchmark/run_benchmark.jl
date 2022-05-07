using Pkg
Pkg.develop("StochasticSynapses")
#include("StochasticSynapses\\benchmark\\benchmarks.jl")
include("benchmarks.jl")
#standard_benchmark_threaded("threads_order_benchmark")
#benchmark_read_write_threaded("threaded_IO_benchmark")

function bench()
    N = 1
    M = 2^20
    p = 10
    vmax = 1.5f0
    @show M N StochasticSynapses.VAR_order
    cells = [Cell() for n in 1:M]
    Cell_cycling(cells, N, -1.5f0, vmax)
    Cell_readout(cells)
end

bench()