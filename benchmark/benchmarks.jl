using BenchmarkTools
using CUDA
using JSON
using StochasticSynapses
using Random
using Base.Threads
using Git
using Dates
using LinearAlgebra

function CellArrayCPU_cycling(cells::CellArrayCPU, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    Random.seed!(1)
    nthreads = @show BLAS.get_num_threads()
    M = cells.M
    p = cells.p
    voltages = repeat([vset vreset], M, N)
    bench = @benchmark begin
                @inbounds begin
                    for v in eachslice($voltages, dims=2)
                    applyVoltage!($cells, v)
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :N=>N, :p=>p, :vset=>vset, :vreset=>vreset, :BLASthreads=>nthreads)
    return cells
end

function CellArrayCPU_readout(cells::CellArrayCPU)
    nthreads = @show BLAS.get_num_threads()
    M = cells.M
    p = cells.p
    # Read repeatedly, results go into readout buffer cells.
    bench = @benchmark Ireadout($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :BLASthreads=>nthreads)
    return cells
end

function CellArrayGPU_cycling(cells::CellArrayGPU, N::Int=2^8, vset=-1.5f0, vreset=1.5f0, device=1)
    CUDA.device!(device)
    Random.seed!(1)
    M = cells.M
    p = cells.p
    voltages = CuArray(repeat([vset vreset], M, N))
    bench = @benchmark begin
                CUDA.@sync begin
                    @inbounds begin
                        for v in eachslice($voltages, dims=2)
                            applyVoltage!($cells, v)
                        end
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :N=>N, :p=>p, :vset=>vset, :vreset=>vreset, :device=>CUDA.name(CUDA.device()))
    return cells
end

function CellArrayGPU_readout(cells::CellArrayGPU, device=1)
    CUDA.device!(device)
    M = cells.M
    p = cells.p
    bench = @benchmark CUDA.@sync Ireadout($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :device=>CUDA.name(CUDA.device()))
    return cells
end

function Cell_init(M::Int)
    nthreads = @show Threads.nthreads()
    p = StochasticSynapses.VAR_order
    bench = @benchmark cells = [Cell() for i in 1:M]
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :nthreads=>nthreads)
end

#function Cell_cycling(cells::Vector{StochasticSynapses.CellState}, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
function Cell_cycling(cells, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    # To set nthreads, Julia needs to be started with Julia -t
    nthreads = @show Threads.nthreads()
    Random.seed!(1)
    M = length(cells)
    p = StochasticSynapses.VAR_order
    voltages = repeat([vset vreset], M, N)
    bench = @benchmark begin
                @inbounds begin
                    @threads for i in eachindex($cells)
                        c = $cells[i]
                        for j in 1:2*$N
                            v = $voltages[i, j]
                            $cells[i] = applyVoltage!(c, v)
                        end
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :N=>N, :p=>p, :vset=>vset, :vreset=>vreset, :nthreads=>nthreads)
    return cells
end

function Cell_readout(cells::Vector{StochasticSynapses.CellState})
    nthreads = @show Threads.nthreads()
    M = length(cells)
    p = StochasticSynapses.VAR_order
    currents = Vector{Float32}(undef, M)
    bench = @benchmark begin
                @inbounds begin
                    @threads for i in eachindex($cells)
                        $currents[i] = Ireadout($cells[i])
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :nthreads=>nthreads)
    return cells
end


function write_benchmark(benchmark, meta...)
    # BenchmarkTools only knows how to write benchmarks to json..
    caller = String(StackTraces.stacktrace()[2].func)
    folder = @__DIR__
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    gitrev = strip(read(`$(git()) rev-parse --short HEAD`, String))
    mkpath(folder)

    buf = IOBuffer()
    BenchmarkTools.save(buf, benchmark)
    j = JSON.parse(String(take!(buf)))

    # Insert metadata into the first dict (VERSIONS)
    for m in meta
        j[1][String(m[1])] = m[2]
    end
    j[1]["gitrev"] = gitrev
    fp = joinpath(folder, "$(timestamp)_$(caller).json")
    open(fp,"w") do f
        JSON.print(f, j)
    end
end

function run_structofarray_benchmarks()
    N = 1
    #p = 10
    vmax = 1.5f0
    device = 1
    CUDA.device!(device)

    for p in [1, 10, 20, 40, 60, 80, 90, 100]
        for M in 2 .^ (8:25)
            @show M N p
            CPUcells = CellArrayCPU(M, p)
            CellArrayCPU_cycling(CPUcells, N, -1.5f0, vmax)
            CellArrayCPU_readout(CPUcells)
            # More likely to run out of memory
            if M*p <= 2^25
                GPUcells = CellArrayGPU(M, p)
                CellArrayGPU_cycling(GPUcells, N, -1.5f0, vmax, device)
                CellArrayGPU_readout(GPUcells, device)
            end
        end
    end
end


function run_arrayofstruct_benchmarks()
    # This version can not take p as input yet
    N = 1
    vmax = 1.5f0
    for M in 2 .^ (8:30)
        @show M N StochasticSynapses.VAR_order
        print("Initializing cells...")
        cells = [Cell() for n in 1:M]
        print("Cycling cells...")
        Cell_cycling(cells, N, -1.5f0, vmax)
        print("Reading out cells...")
        Cell_readout(cells)
    end
end