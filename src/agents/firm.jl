abstract type Firm{T <: Real} end

const ν₀ = defaultdietzϕ * y₀ * CtoCO2^2
Base.@kwdef struct StaticFirm{T} <: Firm{T}
    e₀::T = e₀
    ν::T = ν₀
end

Base.@kwdef struct DynamicFirm{T} <: Firm{T}
    e₀::T = e₀
    ν₀::T = ν₀
    ν::T = ν₀ * 0.05
    ω::T = 5e-2
end

function StaticFirm(t, firm::DynamicFirm)
    StaticFirm(firm.e₀, ν(t, firm))
end

function ν(firm::StaticFirm)
    firm.ν
end
function ν(t, firm::DynamicFirm)
    δ = exp(-firm.ω * t)
    return firm.ν₀ * δ + firm.ν * (1 - δ)
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function c(a, firm::StaticFirm)
    firm.ν * a^2 / 2
end
function c(t, a, firm::DynamicFirm)
    δ = exp(-firm.ω * t)
    return (firm.ν₀ * δ + firm.ν * (1 - δ)) * a^2 / 2
end

function k(a, τ, firm::StaticFirm)
    e(a, firm) * τ + c(a, firm)
end
function k(t, a, τ, firm::DynamicFirm)
    e(a, firm) * τ + c(t, a, firm)
end

function aᶜ(τ, firm::StaticFirm)
    min(τ / ν(firm), firm.e₀)
end
function aᶜ(t, τ, firm::DynamicFirm)
    min(τ / ν(t, firm), firm.e₀)
end

"Best response abatement to tax τ given belief φ."
function aᵇ(τ, φ, τᶜ, firm::StaticFirm)
    min((φ * τᶜ + (1 - φ) * τ) / ν(firm), firm.e₀)
end
function aᵇ(t, τ, φ, τᶜ, firm::DynamicFirm)
    min((φ * τᶜ + (1 - φ) * τ) / ν(t, firm), firm.e₀)
end
