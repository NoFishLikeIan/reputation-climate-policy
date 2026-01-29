Base.@kwdef struct Signal{T <: Real}
	α::T = 1 / (3 * taxfactor)
	σ::T = 1.
end

function μ(τ, a, signal::Signal)
	signal.α * τ - a
end

function L(τ, a, z, signal::Signal, government::Government, firm::Firm)
	w(τ, a, government, firm) + z * μ(τ, a, signal) * (μ(τᶜ, a, signal) - μ(τ, a, signal)) / signal.σ^2
end