abstract type AbstractFirm{T <: Real} end
struct FirmPermanentInvestment{T} <: AbstractFirm{T}
	e₀::T
	ν::T
	κ::T
	β::T
end

struct Firm{T} <: AbstractFirm{T}
	e₀::T
	ν::T
	κ::T
	β::T
	δ::T
end

function Firm(; e₀ = e₀, ν = dietzφ * y₀ * (e₀ * CtoCO₂)^2, κ = baldwinκ, δ = 2e-2,  β = 1 - 1e-3)	
	δ > 0 ? Firm(e₀, ν, κ, β, δ) : FirmPermanentInvestment(e₀, ν, κ, β)
end

function c(φ, firm::AbstractFirm)
	firm.κ * φ + (firm.ν / 2) * φ^2
end
function cᵩ(φ, firm::AbstractFirm)
	firm.κ + firm.ν * φ
end

function f(φ, emissions, δ, e₀)
	max(δ * e₀ + (1 - δ) * emissions - φ, zero(emissions))
end
function f(φ, e, firm::Firm)
	f(φ, e, firm.δ, firm.e₀)
end
function f(φ, e, ::FirmPermanentInvestment)
	max(e - φ, zero(e))
end

function investmentupper(e, firm::Firm)
	firm.δ * firm.e₀ + (1 - firm.δ) * e
end

function investmentupper(e, ::FirmPermanentInvestment)
	e
end

function blissabatement(τᶜ, firm::Firm, signal::Signal)
	@unpack κ, ν, δ = firm
	@unpack μ = signal

	return (τᶜ * μ - κ * δ) / (ν * δ^2)
end

function blissabatement(τᶜ, firm::FirmPermanentInvestment, ::Signal)
	blissabatement(τᶜ, firm)
end
function blissabatement(τᶜ, firm::FirmPermanentInvestment)
	(τᶜ > 0) ? firm.e₀ : zero(firm.e₀)
end
