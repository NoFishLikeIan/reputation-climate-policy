Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1.
    σ::T = σ̂ 
end

μ(τ, signal::Signal) = signal.ϵ * τ