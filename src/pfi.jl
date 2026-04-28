const brent = Optim.Brent()
const constextrap = ConstExtrap()

"Expected firm value before the realization of q."
function updateexantevalue!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    continuationgrid = Grid(exantegrid, pricespace)
    innovationspace, signalweights = signal.space

    V = linear_interp(continuationgrid.nodes, firmvalue.continuation.V; extrap = constextrap)

    indices = CartesianIndices(firmvalue.exante)
    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = reputationspace[j]
        p = logistic(z)
        τ = welfare.P[i, j]
        EV = zero(T)

        for (k, ξ) in enumerate(innovationspace)
            qⁿᶜ = realisedprice(ξ, τ, signal)
            qᶜ = realisedprice(ξ, τᶜ, signal)

            EV += signalweights[k] * (
                (1 - p) * evaluatefirmcontinuationvalue(V, a, z, qⁿᶜ, continuationgrid, τᶜ, firm, signal) +
                p * evaluatefirmcontinuationvalue(V, a, z, qᶜ, continuationgrid, τᶜ, firm, signal)
            )
        end

        firmvalue.exante[i, j] = EV
    end

    return firmvalue
end

function setfirmboundaries!(firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(reputationspace)
        ω = (z - zmin) / (zmax - zmin)

        for (k, q) in enumerate(pricespace)
            v = v̲(a, q, firm) * (1 - ω) + v̄(a, q, τᶜ, firm, signal) * ω
            φ = φ̲(a, q, firm, signal) * (1 - ω) + φ̄(τᶜ, firm, signal) * ω

            firmvalue.continuation.V[i, j, k] = v
            firmvalue.continuation.P[i, j, k] = φ
        end

        firmvalue.exante[i, j] = ψ̄(a, τᶜ, firm, signal) * ω
    end

    return firmvalue
end

function setgovernmentboundaries!(welfare::TW, τᶜ, exantegrid::G, firm::Firm, government::Government, signal::Signal) where {T, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(reputationspace)
        ω = (z - zmin) / (zmax - zmin)

        welfare.V[i, j] = w̲(a, firm, government) * (1 - ω) + w̄(a, τᶜ, firm, government, signal) * ω
        welfare.P[i, j] = τ̲(a, firm, government) * (1 - ω) + τ̄(τᶜ, firm, government) * ω
    end

    return welfare
end

@inline function firmobjective(φ, a, z′, Ψ::LI, exantegrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    a′ = f(φ, a, firm)
    exante = evaluatefirmexantevalue(Ψ, a′, z′, exantegrid, τᶜ, firm, signal)

    return c(φ, firm) + firm.β * exante
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal; φlims = (0., 1.)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    Ψ = linear_interp(exantegrid.nodes, firmvalue.exante; extrap = constextrap)

    indices = CartesianIndices(nextfirmvalue.continuation.V)
    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        a = abatementspace[i]
        z = reputationspace[j]
        q = pricespace[k]
        τ = welfare.P[i, j]
        z′ = z + ℓ(q, τ, τᶜ, signal)

        res = Optim.optimize(φ -> firmobjective(φ, a, z′, Ψ, exantegrid, τᶜ, firm, signal), φlims[1], φlims[2], brent)

        nextfirmvalue.continuation.V[i, j, k] = e(a, firm) * q + Optim.minimum(res)
        nextfirmvalue.continuation.P[i, j, k] = Optim.minimizer(res)
    end

    updateexantevalue!(nextfirmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal)

    return nextfirmvalue
end

function governmentobjective(τ, τᶜ, a, z, Φ::LIΦ, W::LIW, continuationgrid::GE, exantegrid::GA, firm::Firm, government::Government, signal::Signal) where {LIΦ <: FastInterpolations.AbstractInterpolant, LIW <: FastInterpolations.AbstractInterpolant, GE <: Grid{3}, GA <: Grid{2}}
    innovationspace, signalweights = signal.space

    EV = zero(τ)
    @inbounds for (k, ξ) in enumerate(innovationspace)
        q = realisedprice(ξ, τ, signal)
        z′ = z + ℓ(q, τ, τᶜ, signal)
        φ = evaluatefirmpolicy(Φ, a, z, q, continuationgrid, τᶜ, firm, signal)
        a′ = f(φ, a, firm)

        EV += signalweights[k] * (c(φ, firm) + government.β * evaluatewelfarevalue(W, a′, z′, exantegrid, τᶜ, firm, government, signal))
    end

    EV
end
