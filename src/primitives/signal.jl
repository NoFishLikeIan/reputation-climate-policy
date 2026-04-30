struct Signal{T, H <: NTuple{2, AbstractVector{T}}}
    μ::T
    σ::T
    space::H
end

function Signal(μ, σ, n::Int)
    space, weights = gausshermite(n)
    unitaryweights = weights ./ √π

    return Signal(μ, σ, (space, unitaryweights))
end

const sqrt2 = √2

@inline function realisedprice(ξ, τ, signal::Signal)
    signal.μ * τ + sqrt2 * signal.σ * ξ
end

@inline function impliedsignal(q, τ, signal::Signal)
    (q - signal.μ * τ) / (sqrt2 * signal.σ)
end

@inline function ℓ(q, τ, τᶜ, signal::Signal)
    (signal.μ * (τᶜ - τ) / signal.σ^2) * (q - signal.μ * (τ + τᶜ) / 2)
end

