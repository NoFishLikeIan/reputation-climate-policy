Base.@kwdef struct Signal{T <: Real}
    ϵ::T = 1.
    σ::T = 0.00079773 * √252
end

μ(τ, signal::Signal) = signal.ϵ * τ