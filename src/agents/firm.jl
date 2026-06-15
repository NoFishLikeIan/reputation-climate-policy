abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    ν::T = defaultdietzϕ
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function c(a, firm::Firm)
    firm.ν * a^2 / 2
end

function k(a, τ, firm::Firm)
    e(a, firm) * τ + c(a, firm)
end

function a(τ, firm::Firm)
    min(τ / firm.ν, firm.e₀)
end

"Best response abatement to tax τ given belief φ."
function aᵇ(τ, φ, τᶜ, firm::Firm)
    a(φ * τᶜ + (1 - φ) * τ, firm)
end