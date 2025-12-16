Base.@kwdef struct Firm{T <: Real}
    β::T = 1 - 1e-2 # discount factor [-]

    δ::T = 0.05 # depreciation rate [1/year]
    ē::T = 9.4 # emissions [GtC/year]

    κ::T = 2.11 # base investment cost [t$]
    θ::T = 1.0 # marginal investment difficulty
    ν::T = 0.5 # adjustment costs [year]
    α::T = 0.03 # investment effectiveness [1 / t$]
end

function p(a, firm::Firm)
    firm.κ * (1 + firm.θ * a / (1 - a))
end
function p′(a, firm::Firm)
    firm.κ * firm.θ / (1 - a)^2
end

"Firm's abatement investment cost."
function c(a, φ, firm::Firm) 
    p(a, firm) * φ + (firm.ν / 2) * φ^2
end
function cₐ(a, φ, firm)
    p′(a, firm) * φ
end
function cᵩ(φ, firm::Firm)
    firm.ν * φ
end
cᵩ(a, φ, firm) = cᵩ(φ, firm)

"Firm's total emissions."
function emissions(a, firm::Firm)
    firm.ē * (1 - a)
end

"Capital dynamics `aₜ₊₁ = f(aₜ, φₜ)`"
function f(a, φ, firm::Firm)
    (1 - firm.δ) * a + (1 - a) * firm.α * φ
end
function fₐ(φ, firm::Firm)
    1 - firm.δ - firm.α * φ
end
fₐ(a, φ, firm::Firm) = fₐ(φ, firm)

function fᵩ(a, firm::Firm)
    (1 - a) * firm.α
end
fᵩ(a, φ, firm) = fᵩ(a, firm)

function impliciteuler(xₜ, xₜ₊₁, firm::Firm)
    aₜ, φₜ = xₜ
    aₜ₊₁, φₜ₊₁ = xₜ₊₁

    

end