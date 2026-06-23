Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1.
    σ::T = .000814 * √365
end

μ(τ, signal::Signal) = signal.ϵ * τ