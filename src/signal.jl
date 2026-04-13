Base.@kwdef struct Signal{T}
    μ::T = 1.0
    σ::T = 1.0
end

"Computes the logit-drift "
function logitdrift(s, τ, τᶜ, signal::Signal)
    δτ = s - (τ + τᶜ) / 2
    return δτ * (τ - τᶜ) / signal.σ^2
end