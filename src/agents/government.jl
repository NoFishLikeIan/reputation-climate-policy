Base.@kwdef struct Government{T <: Real}
    ξ::T = defaultscc / (e₀ * CtoCO2)
    y₀::T = y₀
    δ::T = 10.0
    r::T = 0.01
end

function d(e, government::Government)
    government.ξ * e^2 / 2
end

function l(a, τ, government::Government, firm::Firm)
    (government.δ / government.y₀) * τ^2 * firm.e₀ * e(a, firm) / 2
end

function w(τ, a, government::Government, firm::Firm)
    government.y₀ * d(e(a, firm), government) + c(a, firm) + l(a, τ, government, firm)
end
