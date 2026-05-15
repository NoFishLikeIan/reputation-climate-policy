Base.@kwdef struct Firm{T <: Real}
    e₀::T = e₀
    ν::T = defaultdietzϕ * y₀ * CtoCO2^2
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function c(a, firm::Firm)
    firm.ν * a^2 / 2
end

function k(a, τ, firm::Firm)
    e(a, firm) * τ + abatementcost(a, firm)
end

function aᶜ(τ, firm::Firm)
    min(τ / firm.ν, firm.e₀)
end

function kᵉ(a, τ, φ, τᶜ, firm::Firm)
    φ * firmcost(a, τᶜ, firm) + (1 - φ) * firmcost(a, τ, firm)
end

"Best response abatement to tax τ given belief φ."
function aᵇ(τ, φ, τᶜ, firm::Firm)
    min((φ * τᶜ + (1 - φ) * τ) / firm.ν, firm.e₀)
end
