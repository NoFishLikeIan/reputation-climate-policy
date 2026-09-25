function governmentvaluedrift(W, a, m, u, τ, ∂ₘW, ∂ₐW, ∂ᵩW, ∂ᵩᵩW, bᵩ, σᵩ, firm::Firm, government::Government, climate::Climate)
    flowcost = transitionflowcost(a, m, u, τ, firm, government, climate)

    return -government.r * W + government.r * flowcost +
        cumulativeemissionsdrift(a, firm) * ∂ₘW +
        u * ∂ₐW + bᵩ * ∂ᵩW + σᵩ^2 * ∂ᵩᵩW / 2
end

struct CommittedTaxPath{TI, T}
    taxpath::TI
    ts::T
end

function (path::CommittedTaxPath{TI, T})(t) where {TI, T}
    t < path.terminal ? path.active(t) : zero(T)
end

Base.eltype(::CommittedTaxPath{TI, T}) where {TI, T} = T

function committedtaxterminal(activeterminal::T, terminalabatement, firm::Firm, government::Government; tolerance = 0.1taxfactor) where T
    activeterminal
end
