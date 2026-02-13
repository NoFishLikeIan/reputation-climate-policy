const τ₀ = 3. * taxfactor

Base.@kwdef struct Signal{T <: Real}
	σ::T = √τ₀ * 0.1
	α::T = τ₀ * 0.05
	ϵ::T = 1e-2
end

function μ(τ, a, signal::Signal)
	signal.ϵ * (τ - signal.α * a)
end

function L(τ, a, z, signal::Signal, government::Government, firm::Firm)
	w(τ, a, government, firm) - z * μ(τ, a, signal) * (μ(τᶜ, a, signal) - μ(τ, a, signal)) / signal.σ^2
end