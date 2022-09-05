#=
Array of cells are contained in a single struct and matrix operations are used throughout
Runs on GPU and CPU using the same code
Data in CellArrayCPU will run on CPU and data in CellArrayGPU will run on GPU
All cells have to be addressed at once for read/write operations
=#

using Random # randn!
using LinearAlgebra # mul!
using NPZ: npzread
using CUDA

########## Define our CPU and GPU types.  Constructor comes much later

struct CellArrayCPU
    M::Int64                     # scalar      (number of cells)
    p::Int64                     # scalar      (order of VAR model)
    X::Array{Float32, 2}         # 4(p+1) × M  (feature history and εₙ for all cells)
    Xbuf::Array{Float32, 2}      # 4(p+1) × M  (buffer to improve efficiency of shift operation)
    xn::Array{Float32, 2}        # 4 × M       (generated normal feature vectors ̂x*ₙ)
    s::Array{Float32, 2}         # 4 × M       (DtD scale vectors)
    y::Array{Float32, 2}         # 4 × M       (scaled feature vector)
    r::Array{Float32, 1}         # M × 1       (device state variables)
    n::Array{UInt32, 1}          # M × 1       (cycle numbers)
    UR::Array{Float32, 1}        # M × 1       (voltage thresholds for reset switching)
    resetPoly::Array{Float32, 2} # M × 3       (polynomial coefficients for reset transitions)
    VARcoefs::Array{Float32, 2}  # 4 × 4(p+1)  (Ainv Ci and Ainv B concatenated)
    γ::Array{Float32, 2}         # 4 × 6       (coefficients for the denormalizing transformation)
    Iread::Array{Float32, 1}     # M × 1       (readout buffer)
    inHRS::BitVector
    inLRS::BitVector
    setMask::BitVector
    resetMask::BitVector
    fullResetMask::BitVector
    partialResetMask::BitVector
    transCalcMask::BitVector
    drawVARMask::BitVector
end

struct CellArrayGPU
    M::Int64
    p::Int64
    X::CuArray{Float32, 2}
    Xbuf::CuArray{Float32, 2}
    xn::CuArray{Float32, 2}
    s::CuArray{Float32, 2}
    y::CuArray{Float32, 2}
    r::CuArray{Float32, 1}
    n::CuArray{UInt32, 1}
    UR::CuArray{Float32, 1}
    resetPoly::CuArray{Float32, 2}
    VARcoefs::CuArray{Float32, 2}
    γ::CuArray{Float32, 2}
    Iread::CuArray{Float32, 1}
    inHRS::CuArray{Bool, 1}
    inLRS::CuArray{Bool, 1}
    setMask::CuArray{Bool, 1}
    resetMask::CuArray{Bool, 1}
    fullResetMask::CuArray{Bool, 1}
    partialResetMask::CuArray{Bool, 1}
    transCalcMask::CuArray{Bool, 1}
    drawVARMask::CuArray{Bool, 1}
end

##########


"""
Evaluate polynomials defined by coeffs at U
(Horner's algorithm)
"""
function polyval(coeffs::AbstractArray{Float32, 2}, U::AbstractArray{Float32})
    acc = zero(U)
    for c in eachslice(coeffs, dims=2)
        acc .= acc .* U .+ c
    end
    acc
end

function polyval(coeffs::AbstractArray{Float32, 1}, U::AbstractArray{Float32})
    acc = zero(U)
    for c in coeffs
        acc .= acc .* U .+ c
    end
    acc
end

"""
Construct combined matrix for the VAR process for any model order, p ≤ 200
"""
function get_VAR_matrix(p::Int64=20)
    orders = sort([parse(Int, split(k, "_")[1][4:end]) for k in keys(params) if startswith(k, "VAR")])
    q = orders[findfirst(orders .>= p)]
    VAR_params = params["VAR$(q)_model_parameters_Lamprey"]
    VAR_L = Array{Float32, 2}(VAR_params[2:5, :])
    VARcoefs = Array{Float32, 2}(hcat(VAR_L ,[VAR_params[6+4*(n-1):6+4*(n-1)+3, :] for n in p:-1:1]...))
    return VARcoefs
end

"""
Transform data from standard normal to the measured distributions
"""
function Γinv(x::AbstractArray{Float32, 2}, γ::AbstractArray{Float32, 2}=γ)
    exp.(polyval(γ, x))
end

const μ0 = Γinv(zeros(Float32, nfeatures, 1))

# Cleaner notation?
# TODO: replace occurences of RHS with LHS
HRS(c) = view(c.y, iHRS, :)
LRS(c) = view(c.y, iLRS, :)
US(c) = -view(c.y, iUS, :) # -view doesn't work, just makes a copy
UR(c) = view(c.y, iUR, :)

######################################### 

function VAR_sample!(c)
    @inbounds begin
        randn!(view(c.X, 1:nfeatures, :))
        mul!(c.xn, c.VARcoefs, c.X) # all random numbers are replaced, so xn is not accurate anymore where drawVARMask == false
        copy!(c.Xbuf, c.X)
        c.X[5:end-4, :] .= ifelse.(c.drawVARMask', @view(c.Xbuf[9:end, :]), @view(c.Xbuf[5:end-4, :]))
        c.X[end-3:end, :] .= ifelse.(c.drawVARMask', c.xn, @view(c.X[end-3:end,:]))
    end
end

"""
Return r such that (1-r) ⋅ LRSpoly + r ⋅ HHRSpoly intersects I,V
used for switching along transition curves
"""
function r(I::AbstractArray{Float32, 1}, V::AbstractArray{Float32, 1})
    IHHRS_V = polyval(HHRSpoly, V)
    ILLRS_V = polyval(LLRSpoly, V)
    @. (I - ILLRS_V) / (IHHRS_V - ILLRS_V)
end


"""
Current as a function of voltage for the cell state

# TODO: wouldn't this be better? problem is just that the orders of the polynomials are different
# polyval(1-r * LLRSpoly + r * HHRSpoly, U)
"""
function I(r::AbstractArray{Float32, 1}, U)
    (1 .- r) .* polyval(LLRSpoly, U) .+ r .* polyval(HHRSpoly, U)
end

I(c, U) = I(c.r, U)

function transitionParabola(x₁::AbstractArray{Float32, 1}, y₁::AbstractArray{Float32, 1}, y₂::AbstractArray{Float32, 1})
    x₂ = Umax
    x₁² = x₁.^2
    x₂² = x₂.^2
    x₁x₂ = x₁.*x₂
    den = x₁² .+ x₂² .- 2x₁x₂
    dy = y₁ .- y₂
    # Use conditional to maintain concavity
    #if dy >= 0
    a = @. dy / den
    b = @. -2x₂ * a
    c = @. (x₁² * y₂ - 2x₁x₂*y₂ + x₂² * y₁) / den
    #=
    else
        a = -dy / den
        b = 2*x₁*dy / den
        c = (x₁² * y₂ - 2*x₁x₂*y₁ + x₂² * y₁) / den
    end
    =#
    return [a b c]
end

###########################################

function CellArrayCPU(M::Int64=2^16, p::Int64=20)
  VARcoefs = get_VAR_matrix(p)
  X = zeros(Float32, nfeatures*(p+1), M)
  Xbuf = similar(X)
  randn!(view(X, 1:nfeatures, :))
  xn = VARcoefs * X
  X[end-3:end, :] .= xn
  s = Γinv(L * randn(Float32, nfeatures, M)) ./ μ0
  y = Γinv(X[end-3:end, :]) .* s
  resetPoly = Array{Float32, 2}(undef, M, 3)
  r0 = r.(y[iHRS, :])
  n = zeros(UInt32, M)
  UR = y[iUR, :]
  Iread = zeros(Float32, M)
  inHRS = trues(M)
  inLRS = falses(M)
  setMask = falses(M)
  resetMask = falses(M)
  fullResetMask = falses(M)
  partialResetMask = falses(M)
  transCalcMask = falses(M)
  drawVARMask = falses(M)

  return CellArrayCPU(M, p, X, Xbuf, xn, s, y, r0, n, UR, resetPoly, VARcoefs, γ, Iread,
                      inHRS, inLRS, setMask, resetMask, fullResetMask, partialResetMask, transCalcMask, drawVARMask)

end

function CellArrayGPU(M::Int64=2^16, p::Int64=20)
   cellsCPU = CellArrayCPU(M, p)
   # kind of a fragile conversion..
   return CellArrayGPU([getfield(cellsCPU, f) for f in fieldnames(CellArrayGPU)]...)
end


"""
Apply voltage array U to the CellArray
if U > UR or if U ≤ US, states will be modified
"""
function applyVoltage!(c, Ua::AbstractArray{Float32, 1})
    ### Create boolean masks for the different conditions
    c.setMask .= .~c.inLRS .& (Ua .≤ US(c))
    @. c.resetMask = ~c.inHRS & (Ua > c.UR)
    @. c.fullResetMask = c.resetMask & (Ua ≥ Umax)
    @. c.partialResetMask = c.resetMask & (Ua < Umax)
    @. c.drawVARMask = c.inLRS & c.resetMask
    @. c.transCalcMask = c.drawVARMask & ~c.fullResetMask

    if any(c.setMask)
        c.r .= ifelse.(c.setMask, r.(LRS(c)), c.r)
        c.inLRS .|= c.setMask
        c.inHRS .= c.inHRS .& .!c.setMask
        c.UR .= ifelse.(c.setMask, UR(c), c.UR)
    end

    if any(c.drawVARMask)
        VAR_sample!(c)
        c.n .+= c.drawVARMask
        c.y .= Γinv(c.X[end-3:end, :], c.γ) .* c.s
    end

    if any(c.transCalcMask)
        x1 = c.UR
        y1 = I(c.r, x1)
        y2 = I(r.(c.y[iHRS, :]), Umax)
        c.resetPoly .= ifelse.(c.transCalcMask, transitionParabola(x1, y1, y2), c.resetPoly)
    end

    if any(c.resetMask)
        c.inLRS .= c.inLRS .& .!c.resetMask
        c.UR .= ifelse.(c.resetMask, Ua, c.UR)
    end

    if any(c.partialResetMask)
        Itrans = polyval(c.resetPoly, Ua)
        c.r .= ifelse.(c.partialResetMask, r(Itrans, Ua), c.r)
    end

    if any(c.fullResetMask)
        c.inHRS .|= c.fullResetMask
        c.r .= ifelse.(c.fullResetMask, r.(HRS(c)), c.r)
    end

    return
end

"""
Simulated ADC measurement including noise
across cells in an array, at a single voltage

(actually mutates c by updating c.Iread..)
"""
function Iread(c, U::Float32=Uread, nbits::Int=4, Imin::Float32=1f-6, Imax::Float32=1f-5, BW::Float32=1f8)
    randn!(c.Iread)
    Inoiseless = I(c.r, U)
    σ_total = @. √(4*kBT*BW*Inoiseless/Uread + abs(2*e*Inoiseless*BW))
    Irange = Imax - Imin
    nlevels = 2^nbits
    q = Irange / nlevels
    @. c.Iread = Inoiseless + c.Iread * σ_total 
    @. c.Iread = clamp(round((c.Iread - Imin) / q), 0, nlevels) * q + Imin
    return c.Iread
end