abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    a₀::T = a₀
    κ::T = defaultdietzϕ * y₀ / realfirmdiscount
    r::T = realfirmdiscount
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function C(a, firm::Firm)
    firm.κ * a^2 / 2
end

function c(a, firm::Firm)
    firm.κ * a
end

c′(firm::Firm) = firm.κ
c′(_, firm::Firm) = c′(firm)