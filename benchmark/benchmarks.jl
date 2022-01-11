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
    # cells = CellArrayCPU(M, p)
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
    #cells = CellArrayCPU(M, p)
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
    #cells = CellArrayGPU(M, p)
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
    #cells = CellArrayGPU(M, p)
    M = cells.M
    p = cells.p
    bench = @benchmark CUDA.@sync Ireadout($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :device=>CUDA.name(CUDA.device()))
    return cells
end

function Cell_cycling(cells::Vector{StochasticSynapses.CellState}, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    # To set nthreads, Julia needs to be started with Julia -t
    nthreads = @show Threads.nthreads()
    Random.seed!(1)
    # cells = [Cell() for n in 1:M]
    M = length(cells)
    p = StochasticSynapses.VAR_order
    # Full set/reset (transition parabola never gets calculated)
    voltages = repeat([vset vreset], M, N)
    #currents = similar(voltages)
    bench = @benchmark begin
                @inbounds begin
                    @threads for i in eachindex($cells)
                        c = $cells[i]
                        for j in 1:2*$N
                            v = $voltages[i, j]
                            applyVoltage!(c, v)
                            #$currents[i, j] = Ireadout(c)
                        end
                    end
                end
            end
    show(b)
    write_benchmark(bench, :M=>M, :N=>N, :p=>p, :vset=>vset, :vreset=>vreset, :nthreads=>nthreads)
    return cells
end

function Cell_readout(cells::Vector{StochasticSynapses.CellState})
    nthreads = @show Threads.nthreads()
    #cells = [Cell() for n in 1:M]
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

function run_benchmarks()
    N = 1
    p = 10
    device = 1
    CUDA.device!(device)
    for M in 2 .^ (8:26)
        @show M N p
        CPUcells = CellArrayCPU(M, p)
        GPUcells = CellArrayGPU(M, p)
        # One cycle should be enough?
        CellArrayCPU_cycling(CPUcells, N, -1.5f0, 1.5f0)
        CellArrayCPU_readout(CPUcells)
        CellArrayGPU_cycling(GPUcells, N, -1.5f0, 1.5f0, device)
        CellArrayGPU_readout(GPUcells, device)
    end
end