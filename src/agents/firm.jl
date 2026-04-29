abstract type AbstractFirm{T <: Real} end
Base.@kwdef struct FirmPermanentInvestment{T} <: AbstractFirm{T}
	e₀::T
	ν::T
	κ::T
	β::T
end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
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

function f(φ, a, firm::Firm)
	min((1 - firm.δ) * a + φ, firm.e₀)
end
function fᵩ(φ, a, firm::Firm)
	(1 - firm.δ) * a + φ < firm.e₀ ? one(φ) : zero(φ)
end

function f(φ, a, firm::FirmPermanentInvestment)
	min(a + φ, firm.e₀)
end
function fᵩ(φ, a, firm::Firm)
	a + φ < firm.e₀ ? one(φ) : zero(φ)
end

function e(a, firm::Firm)
	max(firm.e₀ - a, zero(a))
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