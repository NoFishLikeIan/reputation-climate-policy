Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    δ::T = 10.0
    r::T = 0.01
end

function l(a, τ, government::Government, firm::Firm)
    (government.δ / government.y₀) * τ^2 * firm.e₀ * e(a, firm) / 2
end

function w(m, τ, a, climate::Climate, government::Government, firm::Firm)
    government.y₀ * ( d(m, climate) + c(a, firm) ) + l(a, τ, government, firm)
end