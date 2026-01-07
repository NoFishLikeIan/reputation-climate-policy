Base.@kwdef struct Firm{T <: Real}
    β::T = 1 - 1e-2 # discount factor [-]

    ē::T = 9.4 # emissions [GtC/year]

    κ::T = 0.2 # base investment cost [tEur]
    ω::T = 0.0 # marginal investment difficulty

    α::T = 0.17 # investment effectiveness [1 / tEur]
    ν::T = 0.55 # adjustment costs [year / tEur²]
end

function p(a, firm::Firm)
    δ = firm.ω > 0 ? firm.ω  / (inv(a) - 1) : firm.ω
    return firm.κ * (1 + δ)
end
p′(a, firm::Firm) = ForwardDiff.derivative(a -> p(a, firm), a)

"Firm's abatement investment cost."
function c(a, φ, firm::Firm) 
    p(a, firm) * φ + (firm.ν / 2) * φ^2
end
cₐ(a, φ, firm) = ForwardDiff.derivative(a -> c(a, φ, firm), a)
cᵩ(a, φ, firm) = ForwardDiff.derivative(φ -> c(a, φ, firm), φ)

"Firm's total emissions."
function emissions(a, firm::Firm)
    firm.ē * (1 - a)
end

"Capital dynamics `aₜ₊₁ = f(aₜ, φₜ)`"
function f(a, φ, firm::Firm)
    a + (1 - a) * firm.α * φ
end
fₐ(a, φ, firm::Firm) = ForwardDiff.derivative(a -> f(a, φ, firm), a)
fᵩ(a, φ, firm::Firm) = ForwardDiff.derivative(φ -> f(a, φ, firm), φ)
