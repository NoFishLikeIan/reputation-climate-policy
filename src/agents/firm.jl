abstract type AbstractFirm{T <: Real} end

Base.@kwdef struct Firm{T} <: AbstractFirm{T}
    e₀::T = e₀
    κ::T = defaultdietzϕ * y₀ / realfirmdiscount
    r::T = realfirmdiscount
end

function e(a, firm::Firm)
    firm.e₀ - a
end

function c(a, firm::Firm)
    firm.κ * a
end
