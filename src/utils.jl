"Interpolate vector `V` at `x` assuming `space` range."
function interpolate(V::TV, x, space::TR) where {X, S, TV <: AbstractVector{S}, TR <: AbstractRange{X}}
    if x ≥ space[end] return V[end] end
    if x ≤ space[1] return V[1] end
    
    i = searchsortedfirst(space, x) - 1
    ω = (x - space[i]) / step(space)
    
    return V[i + 1] * ω + V[i] * (1 - ω)
end
function interpolate(V::TV, point, spaces::TR) where {S, X, Y, TV <: AbstractMatrix{S}, TR <: Tuple{AbstractRange{X}, AbstractRange{Y}}}
    x, y = point
    xspace, yspace = spaces

    xclamp = clamp(x, xspace[1], xspace[end])
    yclamp = clamp(y, yspace[1], yspace[end])

    ix = clamp(searchsortedfirst(xspace, xclamp) - 1, 1, length(xspace) - 1)
    iy = clamp(searchsortedfirst(yspace, yclamp) - 1, 1, length(yspace) - 1)

    ωx = (xclamp - xspace[ix]) / step(xspace)
    ωy = (yclamp - yspace[iy]) / step(yspace)

    v00 = V[iy, ix]
    v10 = V[iy, ix + 1]
    v01 = V[iy + 1, ix]
    v11 = V[iy + 1, ix + 1]

    return v00 * (1 - ωx) * (1 - ωy) +
           v10 * ωx * (1 - ωy) +
           v01 * (1 - ωx) * ωy +
           v11 * ωx * ωy
end

const ϕ⁻¹ = (√(5.) - 1.) / 2.
const ϕ⁻² = (3. - √(5.)) / 2.

"Computes the minimum of `x = argmin f(x)` for `x ∈ (a, b)` at `tol` tolernace"
function gssmin(f, a, b; tol = (b - a) / 100)
    Δ = b - a
    
    n = ceil(Int, log(tol / Δ) / log(ϕ⁻¹))
    
    c = a + Δ * ϕ⁻²
    d = a + Δ * ϕ⁻¹

    yc = f(c)
    yd = f(d)

    for _ in 1:n
        if yc < yd
            b = d
            d, yd = c, yc
            Δ *= ϕ⁻¹
            c = a + ϕ⁻² * Δ
            yc = f(c)
        else
            a = c
            c, yc = d, yd
            Δ *= ϕ⁻¹
            d = a + Δ * ϕ⁻¹
            yd = f(d)
        end
    end

    if yc < yd
        return yc, (a + d) / 2
    else
        return yd, (c + b) / 2
    end
end

mutable struct Error{S <: Real}
    relative::S
    absolute::S
end

struct FirmValue{S <: Real}
    V::Matrix{S}
    Φ::Matrix{S}
end

# Error utilities
function zero!(error::Error{S}) where S
    error.absolute = zero(S)
    error.relative = zero(S)
    return error
end
function typemax!(error)
    error.absolute = typemax(S)
    error.relative = typemax(S)
    return error
end

function Base.isless(error::Error{S}, tolerance::Error{S}) where S
    (error.absolute < tolerance.absolute) && (error.relative < tolerance.relative)
end

function errorupdate!(error::Error{S}, x::S, y::S) where S
    a = abs(x - y)
    if a > error.absolute error.absolute = a end
    
    r = abs(x / y) - 1
    if r > error.relative error.relative = r end
end

function Base.show(io::IO, error::Error{S}) where S
    Printf.@printf io "Error(abs=%.2e, rel=%.2e)" error.absolute error.relative
end