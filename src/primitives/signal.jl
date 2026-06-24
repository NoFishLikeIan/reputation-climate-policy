Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1.
    σ::T = σ̂ * √365
end

μ(τ, signal::Signal) = signal.ϵ * τ