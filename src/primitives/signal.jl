Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1.
    σ::T = √365 * 0.0282442
end

μ(τ, signal::Signal) = signal.ϵ * τ