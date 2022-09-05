using StochasticSynapses
import StochasticSynapses: Umax
using Test
using CUDA

@testset "StochasticSynapses.jl" begin
    M = 2^12
    p = 10
    @info "Testing arrays of $M cells with VAR order $p."
    ### Struct of Arrays
    ## CPU
    @testset "Struct of CPU arrays" begin
        # initialize cells
        @info "Initializing struct of CPU arrays"
        @test (global cellsCPU; cellsCPU = CellArrayCPU(M, p); true)
        @info "Checking that cells are in HRS"
        @test all(cellsCPU.inHRS)
        @info "Checking that readout returns reasonable currents"
        @test (Itest = Iread(cellsCPU); all(0f0 .< Itest .< 1f-4))
        @info "Checking that all cells are set after applying -2V"
        @test (applyVoltage!(cellsCPU, fill(-2f0, M)); all(cellsCPU.inLRS))
        @info "Checking that all cells are reset after applying Umax"
        @test (applyVoltage!(cellsCPU, fill(Umax, M)); all(cellsCPU.inHRS))
        # Set some fraction of cells
        @info "Setting a fraction of cells by applying -1V"
        @test (applyVoltage!(cellsCPU, fill(-1f0, M)); true)
        # Partial reset the fraction that were set
        @info "Partially resetting a fraction of cells"
        @test (applyVoltage!(cellsCPU, fill(1f0, M)); true)
        # TODO: continuous IV looping
    end

    ## GPU
    @testset "Struct of GPU arrays" begin
    if CUDA.functional()
        @info "Initializing struct of GPU arrays"
        @test (global cellsGPU; cellsGPU = CellArrayGPU(M, p); true)
        @test all(cellsGPU.inHRS)
        @test (Itest = Iread(cellsGPU); all(0f0 .< Itest .< 1f-4))
        @test (applyVoltage!(cellsGPU, CUDA.fill(-2f0, M)); all(cellsGPU.inLRS))
        @test (applyVoltage!(cellsGPU, CUDA.fill(Umax, M)); all(cellsGPU.inHRS))
        @test (applyVoltage!(cellsGPU, CUDA.fill(-1f0, M)); true)
        @test (applyVoltage!(cellsGPU, CUDA.fill(1f0, M)); true)
    else
        @warn "CUDA is not functional, cannot test."
    end
    end

    @testset "Array of structs" begin
        @info "Initializing array of structs"
        @test (global cells; cells = [Cell() for m in 1:M]; true)
        @test all([c.inHRS for c in cells])
        @test (Itest = Iread.(cells); all(0 .< Itest .< 1f-4))
        @test (applyVoltage!.(cells, fill(-2f0, M)); true)
        @test all([c.inLRS for c in cells])
        @test (applyVoltage!.(cells, fill(1.5f0, M)); true)
        @test (applyVoltage!.(cells, fill(-1f0, M)); true)
        @test (applyVoltage!.(cells, fill(1f0, M)); true)
    end
end
