"DAE for the upper boundary ū over Δm = m̄ - m."
function boundaryupperreversedae(∂ₘΔu₀, Δu, parameters, Δm)
    τᶜ, m̄, climate, government, firm = parameters

    m = m̄ - Δm
    τ = τᶜ(m)
    aᶜ = a(τ, firm)

    wₘ = w(m, 0., aᶜ, climate, government, firm)

    return government.r * (Δu - wₘ) + (firm.e₀ - aᶜ) * ∂ₘΔu₀
end

function gaussianquadraticintegral(a, b, c)
    x = b^2 / 4a

    δ = √(π / a) * exp(c + x) / 2

    return δ * (1 + sign(b) * SF.erf(√x))
end

function u̲(m, climate, government, firm)

    @unpack y₀, r = government
    @unpack γ, ζ = climate
    @unpack e₀ = firm

    β₂ = γ * ζ^2 * e₀^2 / 2
    β₁ = -(r + γ * ζ^2 * e₀ * m)
    β₀ = -γ * ζ^2 * m^2 / 2

    return y₀ * (1 - r * gaussianquadraticintegral(β₂, β₁, β₀))
end