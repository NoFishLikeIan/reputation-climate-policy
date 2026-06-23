abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    ν::T = defaultdietzϕ
    ω::T = 0. # Baseline free abatement
    lresidual₀::T = lresidual₀ # Benchmark residual-exposure loss share
    lretirement₀::T = lretirement₀ # Benchmark accelerated-retirement loss share
    a₀::T = a₀ # Benchmark abatement level
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function c(a, firm::Firm)
    firm.ν * a^2 / 2
end

function k(a, τ, government, firm::Firm)
    e(a, firm) * τ + government.y₀ * c(a, firm)
end

function a(τ, government, firm::Firm)
    τ / (government.y₀ * firm.ν)
end

"Best response abatement to tax τ given belief φ."
function aᵇ(τ, φ, τᶜ, government, firm::Firm)
    a(φ * τᶜ + (1 - φ) * τ, government, firm)
end

function netzeroτ(government, firm::Firm)
    government.y₀ * firm.ν * firm.e₀
end
