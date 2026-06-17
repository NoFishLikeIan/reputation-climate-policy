Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 1e-2
end

function λ(government::Government, firm::Firm)
    2firm.l₀ / (τ₀ * (firm.a₀ - firm.ω * firm.e₀) / government.y₀)^2
end

function l(a, τ, government::Government, firm::Firm)
    if a ≤ firm.ω * firm.e₀
        return zero(a)
    end

    δ = λ(government, firm) / 2government.y₀

    return δ * τ^2 * (a - firm.ω * firm.e₀)^2
end

function w(m, τ, a, climate::Climate, government::Government, firm::Firm)
    government.y₀ * ( d(m, climate) + c(a, firm) ) + l(a, τ, government, firm)
end