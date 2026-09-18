abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    a₀::T = a₀
    A::T = defaultA
    ηᴱ::T = defaultηᴱ
    κ::T = defaultdietzϕ * y₀ / realfirmdiscount
    ξ::T = defaultξ
    r::T = realfirmdiscount
end

function y(n, firm::Firm)
    firm.A * n
end

function wage(τ, firm::Firm)
    firm.A * (1 - firm.ηᴱ * τ)
end
function n(τ, household::Household, firm::Firm)
    n(firm.A * (1 - firm.ηᴱ * τ), household)
end

function n′(τ, household::Household, firm::Firm)
    -firm.ηᴱ * n(τ, household, firm) / (household.φᴸ * (1 - firm.ηᴱ * τ))
end

function e(n, a, firm::Firm)
    firm.ηᴱ * y(n, firm) - a
end

function e′(τ, household::Household, firm::Firm)
    firm.ηᴱ * y(n′(τ, household, firm), firm)
end

function 𝒦(τ, household::Household, firm::Firm)
    -e′(τ, household, firm)
end

function c(a, firm::Firm)
    firm.κ * a
end

c′(firm::Firm) = firm.κ
c′(_, firm::Firm) = c′(firm)

function k(a, u, firm::Firm)
    c(a, firm) * u + firm.ξ * u^2 / 2
end

function adjustmenthorizon(firm::Firm)
    slope = firm.r * c′(firm)

    return (
        firm.r * firm.ξ + √((firm.r * firm.ξ)^2 + 4slope * firm.ξ)
    ) / (2slope)
end