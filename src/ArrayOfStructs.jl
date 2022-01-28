#=
One struct (CellState) per cell
Runs faster on CPU than struct of arrays (CellArrayCPU)
No GPU support
better for sparse operations

uses Float32 like the array version for consistency
but there's very little speed benefit and 32-bit are less convenient to work with

TODO:
⋅ Make model order changeable:
 CellState needs to take model order as a parameter
 We need VAR parameters for all the model orders readily available

⋅ is there a better way to store parameters other than globals with different names?
=#

using StaticArrays: SVector, SMatrix, MVector, MMatrix
using Parameters: @with_kw, @unpack

## Speed benefit from having some of the parameters in a different form (StaticArrays)
const Γcoefs_ = SVector{nfeatures, SVector{Γorder, Float32}}([Γcoefs[i,:] for i in 1:nfeatures])
const L_ = SMatrix{nfeatures, nfeatures, Float32}(params["L"]) * Lscale
const VAR_order = 10
const VAR_params = params["VAR$(VAR_order)_model_parameters_Lamprey"]
const VAR_intercept = SVector{nfeatures, Float32}(VAR_params[1, :])
const VAR_L = SMatrix{nfeatures, nfeatures, Float32}(VAR_params[2:5, :])
const VAR_An = SVector{VAR_order, SMatrix{nfeatures, nfeatures, Float32}}([VAR_params[6+4*(n-1):6+4*(n-1)+3, :] for n in 1:VAR_order])

"""
Transform data from standard normal to the measured distributions
"""
function Γinv(x::SVector{4, Float32})
    exp.(polyval.(Γcoefs_, x))
end

const μ0_ = Γinv(zeros(SVector{4, Float32}))

###########################################

"""
Stores the state of each individual cell
Remembers history of n=order cycles
"""
@with_kw mutable struct CellState
    X::MVector{VAR_order, SVector{nfeatures, Float32}} = [zeros(nfeatures) for n in 1:VAR_order]
    y::MVector{nfeatures, Float32} = zeros(nfeatures)
    s::SVector{nfeatures, Float32} = zeros(nfeatures)
    transitionPoly::Vector{Float32} = zeros(3)
    r::Float32 = 1
    n::UInt64 = 0
    UR::Float32 = 0
    inHRS::Bool = true
    inLRS::Bool = false
end

"""
Return a cell initialized in the HRS state.
"""
function Cell()
    c = CellState()
    c.s = Γinv(SVector{nfeatures, Float32}(L_*randn(nfeatures))) ./ μ0_
    c.X[1] = VAR_sample(c)
    c.y .= Γinv(c.X[1]) .* c.s
    c.r = r(HRS(c))
    c.inHRS = true
    c.inLRS = false
    return c
end

"""
Generate the next VAR vector
"""
function VAR_sample(c::CellState)
    x = VAR_L * randn(Float32, nfeatures) + VAR_intercept
    for i in 1:VAR_order
        j = mod(c.n + 1 - i, VAR_order) + 1
        @inbounds x += VAR_An[i] * c.X[j]
    end
    return x
    #return clamp.(x, -σClip, σClip)
end

# Cleaner notation
HRS(c::CellState) = c.y[iHRS]
LRS(c::CellState) = c.y[iLRS]
US(c::CellState) = -c.y[iUS]
UR(c::CellState) = c.y[iUR]

"""
Return r such that (1-r) ⋅ LRSpoly + r ⋅ HHRSpoly intersects I,V
used for switching along transition curves
"""
function r(I::Float32, V::Float32)
    IHHRS_V = polyval(HHRSpoly, V)
    ILLRS_V = polyval(LLRSpoly, V)
    (I - ILLRS_V) / (IHHRS_V - ILLRS_V)
end


"""
Current as a function of voltage for the cell state
"""
Istate(r::Float32, U::Float32) = (1-r) * polyval(LLRSpoly, U) + r * polyval(HHRSpoly, U)

Istate(c::CellState, U::Float32) = Istate(c.r, U)

IHRS(c::CellState, U::Float32) = Istate(r(HRS(c)), U)

ILRS(c::CellState, U::Float32) = Istate(r(LRS(c)), U)

"""
Return coefficients of the second degree polynomial that connects (x1,y1) to (x2,y2)
with zero derivative at one of the endpoints
"""
function transitionParabola(x₁::Float32, y₁::Float32, y₂::Float32)
    x₂ = Umax
    x₁² = x₁^2
    x₂² = x₂^2
    x₁x₂ = x₁*x₂
    den = x₁² + x₂² - 2x₁x₂
    dy = y₁ - y₂
    # Use conditional to maintain concavity
    #if dy >= 0
        a = dy / den
        b = -2x₂*a
        c = (x₁² * y₂ - 2*x₁x₂*y₂ + x₂² * y₁) / den
    #else
    #    a = -dy / den
    #    b = -2x₁ * a
    #    c = (x₁² * y₂ - 2*x₁x₂*y₁ + x₂² * y₁) / den
    #end
    return (a, b, c)
end

"""
Apply voltage U to a cell
if U > UR or if U ≤ US, it may modify c state
c is also returned
"""
function applyVoltage!(c::CellState, U::Float32)
    if !c.inLRS
        if U < US(c)
            # SET
            c.r = r(LRS(c))
            c.inLRS = true
            c.inHRS = false
            c.UR = UR(c)
            return c
        elseif c.inHRS
            return c
        end
    end

    # By now we know we are not in HRS, so we might reset
    if U > c.UR
        full = U ≥ Umax
        if c.inLRS # First reset
            c.inLRS = false
            # Calculate and store params for next cycle
            x = VAR_sample(c)
            c.n += 1
            @inbounds c.X[mod(c.n, VAR_order) + 1] = x
            if full
                c.y .= Γinv(x) .* c.s
            else
                # We will need the updated transition poly
                x1 = UR(c)
                x2 = Umax
                y1 = Istate(c, x1)
                c.y .= Γinv(x) .* c.s
                y2 = IHRS(c, x2)
                c.transitionPoly .= transitionParabola(x1,y1,y2)
            end
        end

        if full
            # Full RESET
            c.r = r(HRS(c))
            c.inHRS = true
            c.UR = Umax
        else
            # Partial RESET
            Itrans = polyval(c.transitionPoly, U)
            c.r = r(Itrans, U)
            c.UR = U
        end
    end
    return c
end

"""
ADC measurement including noise
"""
function Ireadout(c::CellState, U::Float32=Ureadout, nbits::Int64=4, Imin::Float32=1f-6, Imax::Float32=1f-5, BW::Float32=1f8)
    I = Istate(c, U)

    # Approximation of the thermodynamic noise
    johnson = abs(4*kBT*BW*I/Ureadout)
    shot = abs(2*e*I*BW)

    # Digitization noise
    Irange = Imax - Imin
    nlevels = 2^nbits
    q = Irange / nlevels
    #ADC = q^2 / 12

    # Sample from total noise distribution
    #σ_total = √(johnson + shot + ADC)
    σ_total = √(johnson + shot)
    I = I + randn(Float32) * σ_total

    # Return nearest quantization level?
    return clamp(round((I - Imin) / q), 0, nlevels) * q + Imin
end