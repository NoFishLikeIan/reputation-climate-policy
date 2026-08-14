function firmexpectedtax(φ, τ, τᶜ)
    φ * τᶜ + (1 - φ) * τ
end

function investmentpolicy(q::T, a, firm::Firm) where T
    if iszero(cumulativeemissionsdrift(a, firm))
        return zero(T)
    end

    investment = (q / firm.r - c(a, firm)) / firm.ξ

    return max(investment, zero(T))
end

function firmmarginalvaluedrift(q, a, u, τᵉ, ∂ₘq, ∂ₐq, ∂ᵩᵩq, σᵩ, firm::Firm)
    -firm.r * q + firm.r * (τᵉ - c′(a, firm) * u) +
        cumulativeemissionsdrift(a, firm) * ∂ₘq +
        abatementdrift(u) * ∂ₐq + σᵩ^2 * ∂ᵩᵩq / 2
end
