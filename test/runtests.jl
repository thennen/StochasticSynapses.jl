using StochasticSynapses
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
        cellsCPU = CellArrayCPU(M, p)
        @test all(cellsCPU.inHRS)
        @test (I = Ireadout(cellsCPU); all(0f0 .< I .< 1f-4))
        @test (applyVoltage!(cellsCPU, fill(-2f0, cellsCPU.M)); all(cellsCPU.inLRS))
        @test (applyVoltage!(cellsCPU, fill(1.5f0, cellsCPU.M)); all(cellsCPU.inHRS))
        @test (applyVoltage!(cellsCPU, fill(-1f0, cellsCPU.M)); true)
        @test (applyVoltage!(cellsCPU, fill(1f0, cellsCPU.M)); true)
        # continuous IV looping
    end

    ## GPU
    @testset "Struct of GPU arrays" begin
    if CUDA.functional()
        @info "Initializing struct of GPU arrays"
        cellsGPU = CellArrayGPU(M, p)
        @test all(cellsGPU.inHRS)
        @test (I = Ireadout(cellsGPU); all(0f0 .< I .< 1f-4))
        @test (applyVoltage!(cellsGPU, CUDA.fill(-2f0, cellsGPU.M)); all(cellsGPU.inLRS))
        @test (applyVoltage!(cellsGPU, CUDA.fill(1.5f0, cellsGPU.M)); all(cellsGPU.inHRS))
        # Set some fraction
        @test (applyVoltage!(cellsGPU, CUDA.fill(-1f0, cellsGPU.M)); true)
        # Partial reset some fraction that were set
        @test (applyVoltage!(cellsGPU, CUDA.fill(1f0, cellsGPU.M)); true)
    else
        @info "CUDA is not functional, cannot test."
    end
    end


    # Test that array of structs does exactly the same thing?
    # Or maybe it won't because the random numbers are different..
    @testset "Array of structs" begin
        @info "Initializing array of structs"
        cells = [Cell() for m in 1:M]
        @test all([c.inHRS for c in cells])
        @test (I = Ireadout.(cells); all(0 .< I .< 1f-4))
        @test (applyVoltage!.(cells, fill(-2f0, M)); true)
        @test all([c.inLRS for c in cells])
        @test (applyVoltage!.(cells, fill(1.5f0, M)); true)
        @test (applyVoltage!.(cells, fill(-1f0, M)); true)
        @test (applyVoltage!.(cells, fill(1f0, M)); true)
    end
end
