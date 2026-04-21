struct Signal{T, H <: NTuple{2, AbstractVector{T}}}
    μ::T
    σ::T
    space::H
end

@inline function signalprice(ξ, τ, signal::Signal)
    signal.μ * τ + √2 * signal.σ * ξ
end

function pricespace(signal::Signal, τmin, τmax, n = length(signal.space[1]))
    ξmin, ξmax = extrema(signal.space[1])
    corners = (
        signalprice(ξmin, τmin, signal),
        signalprice(ξmin, τmax, signal),
        signalprice(ξmax, τmin, signal),
        signalprice(ξmax, τmax, signal),
    )

    qmin, qmax = extrema(corners)
    collect(range(qmin, qmax, n))
end

"Computes the logit-drift "
function logitdrift(s, τ, τᶜ, signal::Signal)
    (signal.μ * (τᶜ - τ) / signal.σ^2) * (s - signal.μ * (τ + τᶜ) / 2)
end
