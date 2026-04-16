Base.@kwdef struct Government{T}
	ξ::T = dicescc / e₀ # linear damage coefficient [-]
    y₀::T = y₀ # output/GDP [trillion Eur/year]
	β::T = 0.99
end

# Climate damages
function d(e, government::Government)
	(government.ξ / 2) * e^2
end;