Base.@kwdef struct Government{T <: Real}
    β::T = 1 - 1e-2   # discount factor [-]
    ξ₀::T = 0.035 # linear damage coefficient [-]
    ξ₁::T = 0.0018 # quadratic damage coefficient [1/GtC]
    Y::T = 15.231 # output/GDP [trillion Eur/year]
    tcre::T = 3.52e-4 # transient climate response to cumulative emissions [°C/GtC]
end

# Climate damages
function warming(a, firm::Firm, government::Government)
    firm.ē * (1 - a) * government.tcre
end

function d(a, firm::Firm, government::Government)
    T = warming(a, firm, government)
    return government.ξ₀ * T + government.ξ₁ * T^2
end

function socialcost(a, ϕ, firm::Firm, government::Government)
    c(a, ϕ, firm) + d(a, firm, government) * government.Y
end

function τ(t, τ₀, θ)
    τ₀ + θ * t
end