#=
One struct (CellState) per cell
Runs faster on CPU than struct of arrays (CellArrayCPU)
No GPU support with this data structure

better for sparse operations

uses Float32 like the StructOfArrays version for consistency
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

#const γ_ = SVector{nfeatures, SVector{γorder, Float32}}([γ[i,:] for i in 1:nfeatures])
#const L_ = SMatrix{nfeatures, nfeatures, Float32}(params["L"]) * √a
#const p = 10
#const VAR_params = params["VAR$(p)_model_parameters_Lamprey"]
## const VAR_intercept = SVector{nfeatures, Float32}(VAR_params[1, :])
#const VAR_L = SMatrix{nfeatures, nfeatures, Float32}(VAR_params[2:5, :])
#const VAR_An = SVector{p, SMatrix{nfeatures, nfeatures, Float32}}([VAR_params[6+4*(n-1):6+4*(n-1)+3, :] for n in 1:p])
#const LLRSpoly_ = SVector{2, Float32}(LLRSpoly)
#const HHRSpoly_ = SVector{6, Float32}(HHRSpoly)


#const μ0_ = Γinv(zeros(SVector{nfeatures, Float32}))


"""
Struct of model parameters
Should be able to use different model orders and different polynomial degrees

Might need some kind of generated/parameterized type for that

Then maybe CellState can be based on this type to get the p as well

Could make this work with the CellArrays too?  needs a little different format..
Maybe just need to define the conversion
"""
struct CellParams{nfeatures, p, LLRS_deg, HHRS_deg, γ_deg}
    LLRS::SVector{LLRS_deg, Float32}
    HHRS::SVector{HHRS_deg, Float32}
    γ::SVector{nfeatures, SVector{γ_deg, Float32}}
    #VAR_An::SVector{W, SMatrix{nfeatures, nfeatures, Float32}}
    Ai::SVector{p, SMatrix{nfeatures, nfeatures, Float32}}
    B::SMatrix{nfeatures, nfeatures, Float32}
    # a::Float32 only if L is not already multiplied
    L::SMatrix{nfeatures, nfeatures, Float32}
    Umax::Float32
    # not really parameters, but only needs to be computed once instead of M or M*N times
    μ0::SVector{nfeatures, Float32}
    G_LLRS::Float32
    G_HHRS::Float32
    # U₀?
    # p::Int64 ? already in the size of VAR_An, and in the type parameters
    # parameterize nfeatures?  
end

# Constructors:
# can we use kwargs or will that screw everything up?
# Will sending p to this to choose a file name screw up any type inference whatever
#CellParams(Some file name or artifact name, p (to choose which file))

#CellParams(takes abstract arrays and converts everything to (sometimes nested) StaticArrays)
function CellParams(LLRS::AbstractVector, HHRS::AbstractVector, γ::AbstractMatrix, Ai::AbstractArray,
                    B::AbstractMatrix, L::AbstractMatrix, Umax::Number)
    LLRS_deg = length(LLRS)
    HHRS_deg = length(HHRS)
    LLRS_SV = SVector{LLRS_deg, Float32}(LLRS)
    HHRS_SV = SVector{HHRS_deg, Float32}(HHRS)
    nfeatures, γ_deg = size(γ)
    γ_SV = SVector{nfeatures, SVector{γ_deg, Float32}}([γ[i,:] for i in 1:nfeatures])
    G_LLRS = polyval(LLRS_SV, U₀) / U₀
    G_HHRS = polyval(HHRS_SV, U₀) / U₀
    # Ai is 4 x 4p?  or 4 x 4 x p?  either?
    # TODO: put VAR_L/B in the same 3d array as Ai? then it will be 4 x 4 x p+1
    μ0 = exp.(polyval.(γ_SV, zeros(SVector{nfeatures, Float32})))
    p = size(Ai)[end]
    CellParams(LLRS_SV,
               HHRS_SV,
               γ_SV,
               SVector{p, SMatrix{nfeatures, nfeatures, Float32}}([Ai[:,:,i] for i in 1:p]),
               SMatrix{nfeatures, nfeatures, Float32}(B),
               SMatrix{nfeatures, nfeatures, Float32}(L),
               Float32(Umax),
               μ0,
               G_LLRS,
               G_HHRS
               )
end

# some explicitly typed in default parameters so we don't need any files in the repo
# p=1 model
function makeDefaultParams()
    LLRS = [0.00016680415f0, 0.0f0]
    HHRS = [-1.8915786f-7, -1.4242429f-8, 8.997732f-6, 2.7791862f-6, 3.6814893f-7, 0.0f0]
    γ   =  [-0.00042569192 0.0024233772 0.02133848 0.019419907 0.43818042 11.82464;
            5.1612555f-5 -0.0005500864 -0.0013179692 0.00857628 0.13048464 -0.17826475;
            0.0004994454 0.00028674825 0.0014176102 0.007105724 0.09655975 8.996042; 
            2.7241224f-5 0.00010131096 0.0006995542 0.0002662867 0.067291364 -0.33049172]
    Ai =   reshape([0.07134992 0.02479197 0.060121167 0.0001346942;
                    0.014160523 0.14750661 0.039958093 -0.0034643295;
                    0.027714146 0.0030230533 0.25842136 0.06954774;
                    0.0080544595 0.02229603 0.11907225 0.17059253], (4,4,1))
    B  =   [0.996808     0.0        0.0       0.0;
            0.109837     0.982687   0.0       0.0;
            0.0226788   -0.130763   0.949822  0.0;
            0.00513014   0.0508866  0.213302  0.947827]
    L  =   [1.00169     0.0        0.0       0.0;
            0.116571    0.993346   0.0       0.0;
            0.0415841  -0.123457   0.992424  0.0;
            0.0165302   0.0578356  0.260465  0.963922]
    Umax = 1.5
    #U0 = 0.2
    CellParams(LLRS, HHRS, γ, Ai, B, L, Umax)
end
const defaultParams = makeDefaultParams()

# typeof(defaultParams).parameters

###########################################

"""
Stores the state of each individual cell
Remembers history of p cycles

cellParams should determine p
"""
@with_kw mutable struct CellState{nfeatures, p}
    X::MVector{p, SVector{nfeatures, Float32}} = [zeros(SVector{nfeatures, Float32}) for n in 1:p]
    y::SVector{nfeatures, Float32} = zeros(SVector{nfeatures, Float32})
    s::SVector{nfeatures, Float32} = ones(SVector{nfeatures, Float32})
    transitionPoly::SVector{3, Float32} = zeros(SVector{3, Float32})
    r::Float32 = 1
    n::UInt32 = 0
    UR::Float32 = 0
    inHRS::Bool = true
    inLRS::Bool = false
    params::CellParams{nfeatures, p} = defaultParams #nfeatures, p, LLRS_deg, HHRS_deg, γ_deg
end


"""
Transform data from standard normal to the measured distributions
"""
function Γinv(cell::CellState, x::SVector{nfeatures, Float32})
    exp.(polyval.(cell.params.γ, x))
end

# Can erase LLRS_deg, HHRS_deg, γ_deg type parameters, any consequence?
@with_kw mutable struct CellState3{nfeatures, p, LLRS_deg, HHRS_deg, γ_deg}
    params::CellParams{nfeatures, p, LLRS_deg, HHRS_deg, γ_deg} = defaultParams
    X::MVector{p, SVector{nfeatures, Float32}} = MVector{p, SVector{nfeatures, Float32}}(params.B * randn(SVector{nfeatures, Float32}), (zeros(SVector{nfeatures, Float32}) for n in 2:size(params.Ai)[1]))
    s::SVector{nfeatures, Float32} = exp.(polyval.(params.γ, params.L * randn(SVector{nfeatures, Float32}))) ./ params.μ0
    y::SVector{nfeatures, Float32} = exp.(polyval.(params.γ, X[1])) .* s
    transitionPoly::SVector{3, Float32} = zeros(SVector{3, Float32})
    r::Float32 = 1
    n::UInt32 = 0
    UR::Float32 = 0
    inHRS::Bool = true
    inLRS::Bool = false
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


function Cell3()
    x = VAR_L * randn(SVector{nfeatures, Float32}) # + VAR_intercept
    p = 1
    X::MVector{p, SVector{nfeatures, Float32}}  = [zeros(SVector{nfeatures, Float32}) for n in 1:p]
    X[1] = x
    s = Γinv(L_ * randn(SVector{nfeatures, Float32})) ./ μ0_
    y = Γinv(x) .* s
    r0 = r(y[iHRS])
    transitionPoly = zeros(SVector{3, Float32})
    return CellState3(params=defaultParams, X=X, y=y, s=s, r=r0, transitionPoly=transitionPoly)
end

function Cell4()
    x = VAR_L * randn(SVector{nfeatures, Float32}) # + VAR_intercept
    p = 1
    X::MVector{p, SVector{nfeatures, Float32}}  = [zeros(SVector{nfeatures, Float32}) for n in 1:p]
    X[1] = x
    s = Γinv(L_ * randn(SVector{nfeatures, Float32})) ./ μ0_
    y = Γinv(x) .* s
    r0 = r(y[iHRS])
    transitionPoly = zeros(SVector{3, Float32})
    return CellState4(params=defaultParams, X=X, y=y, s=s, r=r0, transitionPoly=transitionPoly)
end

"""
Generate the next VAR vector
"""
function VAR_sample(c::CellState)
    B, nfeatures
    x = c.params.B * randn(SVector{c.params.nfeatures, Float32}) # + VAR_intercept
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