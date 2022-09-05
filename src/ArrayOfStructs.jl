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

⋅ is there a better way to store parameters other than globals with different names?
=#

using StaticArrays: SVector, SMatrix, MVector, MMatrix
using Parameters: @with_kw, @unpack

## Speed benefit from having some of the parameters in a different form (StaticArrays)
# But StaticArrays might not be beneficial for the GPU version (to be tested)
# For now, just create new globals with _ appended
const γ_ = SVector{nfeatures, SVector{γorder, Float32}}([γ[i,:] for i in 1:nfeatures])
const L_ = SMatrix{nfeatures, nfeatures, Float32}(params["L"]) * √a
const p = 10
const VAR_params = params["VAR$(p)_model_parameters_Lamprey"]
# const VAR_intercept = SVector{nfeatures, Float32}(VAR_params[1, :])
const VAR_L = SMatrix{nfeatures, nfeatures, Float32}(VAR_params[2:5, :])
const VAR_An = SVector{p, SMatrix{nfeatures, nfeatures, Float32}}([VAR_params[6+4*(n-1):6+4*(n-1)+3, :] for n in 1:p])
const LLRSpoly_ = SVector{2, Float32}(LLRSpoly)
const HHRSpoly_ = SVector{6, Float32}(HHRSpoly)

"""
Transform data from standard normal to the measured distributions
"""
function Γinv(x::SVector{4, Float32})
    exp.(polyval.(γ_, x))
end

const μ0_ = Γinv(zeros(SVector{4, Float32}))


"""
Struct of model parameters
Should be able to use different model orders and different polynomial degrees

Might need some kind of generated/parameterized type for that

Then maybe CellState can be based on this type to get the p as well

Could make this work with the CellArrays too?  needs a little different format..
Maybe just need to define the conversion
"""
@with_kw struct CellParams
    γ::SVector{nfeatures, SVector{γorder, Float32}}
    a::Float32 = 1.0f0
    L::SMatrix{nfeatures, nfeatures, Float32}
    p::Int64 = 10
    VAR_L::SMatrix{nfeatures, nfeatures, Float32}
    VAR_An::SVector{}
    LLRS::SVector{}
    HHRS::SVector{}
    Umax::Float32 = 1.5f0 #?
end

###########################################

"""
Stores the state of each individual cell
Remembers history of n=order cycles
"""
@with_kw mutable struct CellState
    X::MVector{p, SVector{nfeatures, Float32}} = [zeros(SVector{nfeatures, Float32}) for n in 1:p]
    y::SVector{nfeatures, Float32} = zeros(SVector{nfeatures, Float32})
    s::SVector{nfeatures, Float32} = ones(SVector{nfeatures, Float32})
    transitionPoly::SVector{3, Float32} = zeros(SVector{3, Float32})
    r::Float32 = 1
    n::UInt32 = 0
    UR::Float32 = 0
    inHRS::Bool = true
    inLRS::Bool = false
    # params::CellParams = somedefault
end



"""
Return a cell initialized in the HRS state.
"""
function Cell()
    x = VAR_L * randn(SVector{nfeatures, Float32}) # + VAR_intercept
    X::MVector{p, SVector{nfeatures, Float32}}  = [zeros(SVector{nfeatures, Float32}) for n in 1:p]
    X[1] = x
    s = Γinv(L_ * randn(SVector{nfeatures, Float32})) ./ μ0_
    y = Γinv(x) .* s
    r0 = r(y[iHRS])
    transitionPoly = zeros(SVector{3, Float32})
    c = CellState(X=X, y=y, s=s, r=r0)
    return c
end

"""
Generate the next VAR vector
"""
function VAR_sample(c::CellState)
    x = VAR_L * randn(SVector{nfeatures, Float32}) # + VAR_intercept
    for i in 1:p
        j = mod(c.n + 1 - i, p) + 1
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
    IHHRS_V = polyval(HHRSpoly_, V)
    ILLRS_V = polyval(LLRSpoly_, V)
    (I - ILLRS_V) / (IHHRS_V - ILLRS_V)
end


"""
Current as a function of voltage for the current cell state (r)
"""
I(r::Float32, U::Float32) = (1-r) * polyval(LLRSpoly_, U) + r * polyval(HHRSpoly_, U)
I(c::CellState, U::Float32) = I(c.r, U)

IHRS(c::CellState, U::Float32) = I(r(HRS(c)), U)
ILRS(c::CellState, U::Float32) = I(r(LRS(c)), U)

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
    #return (a, b, c)
    return SVector{3, Float32}(a, b, c)
end

"""
Apply voltage U to a cell
if U > UR or if U ≤ US, it may modify c state
c is also returned
"""
function applyVoltage!(c::CellState, Ua::Float32)
    if !c.inLRS
        if Ua < US(c)
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
    if Ua > c.UR
        full = Ua ≥ Umax
        if c.inLRS # First reset
            c.inLRS = false
            # Calculate and store params for next cycle
            x = VAR_sample(c)
            c.n += 1
            @inbounds c.X[mod(c.n, p) + 1] = x
            if full
                c.y = Γinv(x) .* c.s
            else
                # We will need the updated transition poly
                x1 = UR(c)
                x2 = Umax
                y1 = I(c, x1)
                c.y = Γinv(x) .* c.s
                y2 = IHRS(c, x2)
                c.transitionPoly = transitionParabola(x1,y1,y2)
            end
        end

        if full
            # Full RESET
            c.r = r(HRS(c))
            c.inHRS = true
            c.UR = Umax
        else
            # Partial RESET
            Itrans = polyval(c.transitionPoly, Ua)
            c.r = r(Itrans, Ua)
            c.UR = Ua
        end
    end
    return c
end

"""
ADC measurement including noise
"""
function Iread(c::CellState, U::Float32=Uread, nbits::Int64=4, Imin::Float32=1f-6, Imax::Float32=1f-5, BW::Float32=1f8)
    Inoiseless = I(c, U)
    # Approximation of the thermodynamic noise
    johnson = abs(4*kBT*BW*Inoiseless/Uread)
    shot = abs(2*e*Inoiseless*BW)
    # Digitization noise
    Irange = Imax - Imin
    nlevels = 2^nbits
    q = Irange / nlevels
    #ADC = q^2 / 12
    # Sample from total noise distribution
    #σ_total = √(johnson + shot + ADC)
    σ_total = √(johnson + shot)
    Iwithnoise = Inoiseless + randn(Float32) * σ_total
    # Return nearest quantization level?
    return clamp(round((Iwithnoise - Imin) / q), 0, nlevels) * q + Imin
end