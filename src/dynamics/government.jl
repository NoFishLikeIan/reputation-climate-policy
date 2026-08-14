function transitionflowcost(a, m, u, τ, firm::Firm, government::Government, climate::Climate)
    government.y₀ * d(m, climate) + l(τ, government) + investmentcost(a, u, firm)
end

function governmentvaluedrift(W, a, m, u, τ, ∂ₘW, ∂ₐW, ∂ᵩW, ∂ᵩᵩW, bᵩ, σᵩ, firm::Firm, government::Government, climate::Climate)
    flowcost = transitionflowcost(a, m, u, τ, firm, government, climate)

    return -government.r * W + government.r * flowcost +
        cumulativeemissionsdrift(a, firm) * ∂ₘW +
        abatementdrift(u) * ∂ₐW + bᵩ * ∂ᵩW + σᵩ^2 * ∂ᵩᵩW / 2
end

function cumulativeemissionscostatedrift(λₘ, m, government::Government, climate::Climate)
    government.r * (λₘ - government.y₀ * d′(m, climate))
end

function abatementcostatedrift(λₘ, λₐ, λᵤ, a, u, firm::Firm, government::Government)
    government.r * λₐ - government.r * c′(a, firm) * u + λₘ -
        firm.r * c′(a, firm) * λᵤ / firm.ξ
end

function investmentratecostatedrift(λₐ, λᵤ, a, u, firm::Firm, government::Government)
    (government.r - firm.r) * λᵤ -
        government.r * (c(a, firm) + firm.ξ * u) - λₐ
end

function annualisedcostdrift(P, flowcost, government::Government)
    government.r * (P - flowcost)
end

function committedtailtax(t, ā, firm::Firm, government::Government)
    if !iszero(cumulativeemissionsdrift(ā, firm))
        initialtax = (2firm.r - government.r) * c(ā, firm)

        return initialtax * exp(-(firm.r - government.r) * t)
    end

    return zero(ā)
end

struct CommittedTaxPath{TI, T}
    active::TI
    activeterminal::T
    terminal::T
    tailtax::T
    taildecay::T

    function CommittedTaxPath(active::TI, activeterminal::T, terminal, terminalabatement, firm::Firm, government::Government) where {TI, T}
        tailtax = committedtailtax(zero(T), terminalabatement, firm, government)
        taildecay = firm.r - government.r

        return new{TI, T}(active, activeterminal, terminal, tailtax, taildecay)
    end
end

function (path::CommittedTaxPath{TI, T})(t) where {TI, T}
    if t < path.activeterminal
        return path.active(t)
    elseif path.activeterminal ≤ t ≤ path.terminal
        decay = exp(-path.taildecay * (t - path.activeterminal))
        return path.tailtax * decay
    else
        return zero(T)
    end
end

Base.eltype(::CommittedTaxPath{TI, T}) where {TI, T} = T

function committedtaxterminal(activeterminal::T, terminalabatement, firm::Firm, government::Government; tolerance = 0.1taxfactor) where T
    tailtax = committedtailtax(zero(T), terminalabatement, firm, government)

    if tailtax ≤ tolerance
        return activeterminal
    end

    taildecay = firm.r - government.r

    return activeterminal + log(tailtax / tolerance) / taildecay
end
