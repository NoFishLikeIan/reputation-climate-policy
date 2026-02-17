abstract type AbstractFirm{T <: Number} end

struct InelasticEnergyFirm{T} <: AbstractFirm{T}
	e₀::T # emissions [GtC/year]
	ν::T # adjustment costs [year / tEur²]
end
struct EnergyFirm{T} <: AbstractFirm{T}
	e₀::T # emissions [GtC/year]
	ν::T # adjustment costs [year / tEur²]
	δ::T # Elasticity of energy market
end

function AbstractFirm(e₀ = e₀, ν = dietzφ * y₀ * (e₀ * CtoCO₂)^2, δ = 1e-1)
	if isfinite(δ)
		return Firm(e₀, ν, δ)
	else
		return InelasticEnergyFirm(e₀, ν)
	end
end

function c(a, firm::F) where F <: AbstractFirm
	firm.ν * a^2 / 2
end

function e(a, firm::F) where F <: AbstractFirm
	(1 - a) * firm.e₀
end

function k(a, τ, firm::F) where F <: AbstractFirm
	e(a, firm) * τ + c(a, firm)
end

function aᶜ(τ, firm::F) where F <: AbstractFirm
	min((firm.e₀ / firm.ν) * τ, 1)
end;

function l(a, τ, firm::EnergyFirm)
	(τ * e(a, firm))^2 / firm.δ
end;

function l(_, _, ::InelasticEnergyFirm{T}) where T <: Number
	zero(T)
end;