Base.@kwdef struct Government{T}
	ξ::T = dicescc / e₀ # linear damage coefficient [-]
    y₀::T = y₀ # output/GDP [trillion Eur/year]
	r::T = 0.05
end

# Climate damages
function d(e, gov::Government)
	(gov.ξ / 2) * e^2
end;

function w(τ, a, government::Government, firm::Firm) 
	government.y₀ * d(e(a, firm), government) + c(a, firm) + l(a, τ, firm);
end