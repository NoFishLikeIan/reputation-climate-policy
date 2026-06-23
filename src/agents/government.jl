Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 1e-2
end

function R(a, τ, firm::Firm)
    τ * √(firm.e₀ * e(a, firm))
end

function ρ(a, τ, government::Government, firm::Firm)
    R(a, τ, firm) / government.y₀
end

function δ(government::Government, firm::Firm)
    2firm.l₀ / ρ(firm.a₀, τ₀, government, firm)^2
end

function strandedshare(δ, government::Government, firm::Firm)
    δ * ρ(firm.a₀, τ₀, government, firm)^2 / 2
end

function l(a, τ, government::Government, firm::Firm)
    δ(government, firm) * τ^2 * firm.e₀ * e(a, firm) / (2government.y₀)
end

function w(m, τ, a, climate::Climate, government::Government, firm::Firm)
    government.y₀ * ( d(m, climate) + c(a, firm) ) + l(a, τ, government, firm)
end
