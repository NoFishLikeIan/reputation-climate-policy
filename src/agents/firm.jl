abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    a₀::T = a₀
    A::T = defaultA
    η::T = defaultηᴱ
    κ::T = defaultdietzϕ * y₀ / realfirmdiscount
    ξ::T = defaultξ
end

function y(n, firm::Firm)
    n * firm.A
end

function ω(τ, firm::Firm)
    firm.A * (1 - firm.η * τ)
end
function ω′(_, firm::Firm)
    -firm.A * firm.η
end

function e(n, a, firm::Firm)
    firm.η * y(n, firm) - a
end

function e′(τ, household::AbstractHousehold, firm::Firm)
    firm.η * y(n′(τ, household, firm), firm)
end

function 𝒦(τ, household::AbstractHousehold, firm::Firm)
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
    slope = household.r * c′(firm)

    return (
        household.r * firm.ξ + √((household.r * firm.ξ)^2 + 4slope * firm.ξ)
    ) / (2slope)
end

function δ(firm::Firm)
    @unpack r, κ, ξ = firm

    (√(r^2 + 4r * κ / ξ) - r) / 2
end

function α(q, a, firm::Firm)
    da = ((q / household.r) - c(a, firm)) / firm.ξ

    return max(da, 0)
end