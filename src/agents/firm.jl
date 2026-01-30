abstract type Firm{T <: Real} end

Base.@kwdef struct StaticFirm{T} <: Firm{T}
	e₀::T = e₀ # emissions [GtC/year]
	ν::T = dietzφ * y₀ * (e₀ * CtoCO₂)^2 # adjustment costs [year / tEur²]
	δ::T = 0.025 # [.] depreciation of abatemnet
end

function c(φ, firm::F) where F <: Firm
	(firm.ν / 2) * φ^2
end

function totalcosts(φ, a, τ, firm::F) where F <: Firm
	e(a, firm) * τ + c(φ, firm)
end

function f(φ, a, firm::F) where F <: Firm
	(1 - firm.δ) * a + φ
end

function e(a, firm::StaticFirm)
	firm.e₀ - a
end
