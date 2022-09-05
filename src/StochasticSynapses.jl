module StochasticSynapses

export applyVoltage!, Iread, CellArrayCPU, CellArrayGPU, Cell, HRS, LRS, UR, US, I, Umax, Uread

using NPZ: npzread

function polyval(coeffs::AbstractVector{Float32}, U::Float32)
    acc = zero(U)
    for c in coeffs
        acc = acc * U + c
    end
    acc
end

const Uread = Float32(0.2) # Default voltage to perform readouts
const Umax = Float32(1.5)     # The highest applied voltage in RESET direction during the experiment.
# const σClip = Float32(3.5)
const e = Float32(1.602176634e-19)
const kBT = Float32(1.380649e-23 * 300)

# Load model parameters from disk
const params = npzread(joinpath(@__DIR__, "model_parameters.npz"))
const HHRSpoly = Vector{Float32}(params["HHRS"])
const LLRSpoly = Vector{Float32}(params["LLRS"])
const U₀ = Float32(0.2)
const G_HHRS = polyval(HHRSpoly, U₀) / U₀
const G_LLRS = polyval(LLRSpoly, U₀) / U₀
const γ = Array{Float32}(params["polynomial_flows"])
const nfeatures, γorder = size(γ)
const iHRS, iUS, iLRS, iUR = (Int64(i) for i in 1:nfeatures)
const a = Float32(1)
const L = Array{Float32}(params["L"] * √a) # Σ = LL*

"""
Return r such that (1-r) ⋅ LRSpoly + r ⋅ HHRSpoly has static resistance R = U₀/I(U₀)
"""
function r(R::Float32)
    (G_LLRS - 1/R) / (G_LLRS - G_HHRS)
end

include("StructOfArrays.jl")
include("ArrayOfStructs.jl")

end
