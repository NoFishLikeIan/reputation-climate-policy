Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 0.1
    σ::T = 0.20666
end

μ(τ, signal::Signal) = signal.ϵ * τ