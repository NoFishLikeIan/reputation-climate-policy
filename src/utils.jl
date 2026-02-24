"Interpolate vector `V` at `x` assuming `space` range."
function interpolate(V::TV, x::X, space::TR) where {X, S, TV <: AbstractVector{S}, TR <: AbstractVector{X}}
    if length(V) != length(space)
        error("Length of V ($(length(V))) does not match length of space ($(length(space)))")
    end
    
    # Clamp outside boundaries
    if x ≤ space[1]
        return V[1]
    elseif x ≥ space[end]
        return V[end]
    end

    # Linear interpolation inside
    i = searchsortedfirst(space, x) - 1
    x₀ = space[i]
    x₁ = space[i + 1]
    ω = (x - x₀) / (x₁ - x₀)
    return V[i] * (1 - ω) + V[i + 1] * ω
end
function interpolate(V::TV, xs::TX, space::TR) where {X, S, TV <: AbstractVector{S}, TX <: AbstractVector{X}, TR <: AbstractVector{X}}
    map(x -> interpolate(V, x, space), xs)
end

function interpolate(V::TV, point, spaces::TR) where {S, X, Y, TV <: AbstractMatrix{S}, TR <: Tuple{AbstractVector{X}, AbstractVector{Y}}}
    x, y = point
    xspace, yspace = spaces
    # Clamp in y to edge columns if outside
    if y ≤ yspace[1]
        edge = @view V[:, 1]
        return interpolate(edge, x, xspace)
    elseif y ≥ yspace[end]
        edge = @view V[:, end]
        return interpolate(edge, x, xspace)
    end

    # Bilinear interpolation inside
    j = searchsortedfirst(yspace, y) - 1
    y₀ = yspace[j]
    y₁ = yspace[j + 1]
    lower = @view V[:, j]
    upper = @view V[:, j + 1]
    f₀ = interpolate(lower, x, xspace)
    f₁ = interpolate(upper, x, xspace)
    ω = (y - y₀) / (y₁ - y₀)
    return f₀ * (1 - ω) + f₁ * ω
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