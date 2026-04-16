Base.@kwdef struct Firm{T}
	e₀::T = e₀ # emissions [GtC/year]
	ν::T = dietzφ * y₀ * (e₀ * CtoCO₂)^2 # adjustment costs [year / tEur²]
	κ::T = 0.11 # Baldwin et al. TODO: Double check
	δ::T = 0.025 # [.] depreciation of abatemnet
	β::T = 0.99
end

function c(φ, firm::Firm)
	firm.κ * φ + (firm.ν / 2) * φ^2
end
function cᵩ(φ, firm::Firm)
	firm.κ + firm.ν * φ
end

function f(φ, a, firm::Firm)
	(1 - firm.δ) * a + φ
end
function fᵩ(φ, _, _::Firm)
	one(φ)
end

function e(a, firm::Firm)
	firm.e₀ - a
end

function blissabatement(τᶜ, firm::Firm, signal::Signal)
	@unpack κ, ν, δ = firm
	@unpack μ = signal

	return (τᶜ * μ - κ * δ) / (ν * δ^2)
end