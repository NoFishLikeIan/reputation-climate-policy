const matchscc = 66 / 100

Base.@kwdef struct Government{T <: Real}
    β::T = 1 - 1e-2   # discount factor [-]
    ξ::T = matchscc / 4.2660429 # linear damage coefficient [-]
    y₀::T = 15.231 # GDP [trillion Eur/year]
end

# Climate damages
function d(a, firm::Firm, government::Government)
    (government.ξ / 2) * (1 - a)^2 * firm.e₀^2
end

function socialcost(a, ϕ, firm::Firm, government::Government)
    c(a, ϕ, firm) + d(a, firm, government) * government.y₀
end

function τ(t, τ₀, θ)
    τ₀ * exp(θ * t)
end