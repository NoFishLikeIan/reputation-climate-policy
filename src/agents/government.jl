Base.@kwdef struct Government{T <: Real}
    β::T = 1 - 1e-2   # discount factor [-]
    ξ₀::T = 0.035 # linear damage coefficient [-]
    ξ₁::T = 0.0018 # quadratic damage coefficient [1/GtC]
    Y::T = 78.0 # output/GDP [trillion USD/year]
    tcre::T = 0.45e-3 # transient climate response to cumulative emissions [°C/GtC]
end

# Climate damages
function d(a, firm::Firm, government::Government)
    warming = firm.ē * (1 - a) * government.tcre
    return government.ξ₀ * warming + government.ξ₁ * warming^2
end

function socialcost(a, ϕ, firm::Firm, government::Government)
    c(a, ϕ, firm) + d(a, firm, government) * government.Y
end

function τ(t, τ₀, θ)
    τ₀ * exp(θ * t)
end