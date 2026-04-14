struct Signal{T, H <: NTuple{2, AbstractVector{T}}}
    μ::T
    σ::T
    space::H
end

"Computes the logit-drift "
function logitdrift(s, τ, τᶜ, signal::Signal)
    (signal.μ * (τᶜ - τ) / signal.σ^2) * (s - signal.μ * (τ + τᶜ) / 2)
end