Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1e-2
    σ::T = √τ₀
end

μ(τ, signal::Signal) = signal.ϵ * τ