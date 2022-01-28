module StochasticSynapses

export applyVoltage!, Ireadout, CellArrayCPU, CellArrayGPU, Cell

using NPZ: npzread

function polyval(coeffs::AbstractVector{Float32}, U::Float32)
    acc = zero(U)
    for c in coeffs
        acc = acc * U + c
    end
    acc
end

const Ureadout = Float32(0.2)
# The highest applied voltage in RESET direction during the experiment.
const Umax = Float32(1.5)
# const σClip = Float32(3.5)
const e = Float32(1.602176634e-19)
const kBT = Float32(1.380649e-23 * 300)

# Load model parameters from disk
const params = npzread(joinpath(@__DIR__, "model_parameters.npz"))
const HHRSpoly = Vector{Float32}(params["HHRS"])
const LLRSpoly = Vector{Float32}(params["LLRS"])
const Ustatic = Float32(0.2)
const G_HHRS = polyval(HHRSpoly, Ustatic) / Ustatic
const G_LLRS = polyval(LLRSpoly, Ustatic) / Ustatic
const Γcoefs = Array{Float32}(params["polynomial_flows"])
const nfeatures = size(Γcoefs)[1]
const Γorder = size(Γcoefs)[2]
const iHRS, iUS, iLRS, iUR = (Int64(i) for i in 1:nfeatures)
const Lscale = Float32(1)
const L = Array{Float32}(params["L"] * Lscale)

"""
Return r such that (1-r) ⋅ LRSpoly + r ⋅ HHRSpoly has static resistance R = Ustatic/I(Ustatic)
"""
function r(R::Float32)
    (G_LLRS - 1/R) / (G_LLRS - G_HHRS)
end

include("StructOfArrays.jl")
include("ArrayOfStructs.jl")

end
