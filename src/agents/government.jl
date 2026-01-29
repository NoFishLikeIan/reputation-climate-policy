Base.@kwdef struct Government{T}
	ξ::T = dicescc / e₀ # linear damage coefficient [-]
    y₀::T = y₀ # output/GDP [trillion Eur/year]
	δ::T = 0.2 * y₀
	ρ::T = 0.05
end

# Climate damages
function d(e, gov::Government)
	(gov.ξ / 2) * e^2
end;

function l(τ, gov::Government)
	(gov.δ / 2) * τ^2
end;

function w(τ, a, gov::Government, firm::Firm)
    gov.y₀ * d(e(a, firm), gov) + c(a, firm) + l(τ, gov)
end

function wᶜ(τ, gov::Government, firm::Firm)
	w(τ, aᶜ(τ, firm), gov, firm)
end