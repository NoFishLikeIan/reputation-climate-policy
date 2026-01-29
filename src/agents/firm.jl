Base.@kwdef struct Firm{T}
	e₀::T = e₀ # emissions [GtC/year]
	ν::T = dietzφ * y₀ * (e₀ * CtoCO₂)^2 # adjustment costs [year / tEur²]
end

function c(a, firm::Firm)
	firm.ν * a^2 / 2
end

function e(a, firm::Firm)
	(1 - a) * firm.e₀
end

function k(a, τ, firm::Firm)
	e(a, firm) * τ + c(a, firm)
end

function aᶜ(τ, firm::Firm)
	min((firm.e₀ / firm.ν) * τ, 1)
end;