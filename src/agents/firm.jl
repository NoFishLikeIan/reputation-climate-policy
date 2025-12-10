Base.@kwdef struct Firm{T <: Real}
    β::T = 1 - 1e-2 # discount factor [-]
    ē::T = 9.4 # emissions [GtC/year]
    κ::T = 2.11 / 2 # investment cost parameter [USD]
    b::T = 2.8 # Steepness of MAC
    δ::T = 0.04 # depreciation rate [1/year]
end

"Firm's abatemnet investments cost."
function k(a, ϕ, firm::Firm) 
    (firm.κ / 2) * (1 + a)^firm.b * ϕ^2
end

"Firm's total emissions."
function emissions(a, firm::Firm)
    firm.ē * (1 - a)
end

"Capital dynamics `aₜ₊₁ = f(aₜ, ϕₜ)`"
function f(a, ϕ, firm::Firm)
    (1 - firm.δ) * a + ϕ
end

"Total costs of the energy firm"
function c(a, ϕ, τ, firm::Firm)
    k(a, ϕ, firm) + τ * emissions(a, firm)
end