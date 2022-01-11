using BenchmarkTools
using CUDA
using StochasticSynapses
using Random
using Base.Threads
using Git
using Dates
using LinearAlgebra

function CellArrayCPU_cycling(M::Int=2^12, N::Int=2^8, p::Int=2^4, vset=-1.5f0, vreset=1.5f0)
    Random.seed!(1)
    nthreads = @show BLAS.get_num_threads()
    cells = CellArrayCPU(M, p)
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

function CellArrayCPU_readout(M::Int=2^12, p::Int=2^4)
    nthreads = @show BLAS.get_num_threads()
    cells = CellArrayCPU(M, p)
    # Read repeatedly, results go into readout buffer cells.
    bench = @benchmark Ireadout($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :BLASthreads=>nthreads)
    return cells
end

function CellArrayGPU_cycling(M::Int=2^12, N::Int=2^8, p::Int=2^4, vset=-1.5f0, vreset=1.5f0, device=1)
    CUDA.device!(device)
    Random.seed!(1)
    cells = CellArrayGPU(M, p)
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

function CellArrayGPU_readout(M::Int=2^12, p::Int=2^4, device=1)
    CUDA.device!(device)
    cells = CellArrayGPU(M, p)
    bench = @benchmark CUDA.@sync Ireadout($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :p=>p, :device=>CUDA.name(CUDA.device()))
    return cells
end

function Cell_cycling(M::Int=2^12, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    # To set nthreads, Julia needs to be started with Julia -t
    nthreads = @show Threads.nthreads()
    Random.seed!(1)
    cells = [Cell() for n in 1:M]
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
    p = StochasticSynapses.VAR_order
    write_benchmark(bench, :M=>M, :N=>N, :p=>p, :vset=>vset, :vreset=>vreset, :nthreads=>nthreads)
    return cells
end

function Cell_readout(M::Int=2^12)
    nthreads = @show Threads.nthreads()
    cells = [Cell() for n in 1:M]
    currents = Vector{Float32}(undef, M)
    bench = @benchmark begin
                @inbounds begin
                    @threads for i in eachindex($cells)
                        $currents[i] = Ireadout($cells[i])
                    end
                end
            end
    display(bench)
    p = StochasticSynapses.VAR_order
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
    fp = joinpath(folder, "$(timestamp)_$(caller)_gitrev_$(gitrev).json")
    open(fp,"w") do f
        JSON.print(f, j)
    end
end