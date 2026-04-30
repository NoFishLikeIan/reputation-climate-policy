const brent = Optim.Brent()
const constextrap = ConstExtrap()

"Expected firm value before the realization of q."
function updateexantevalue!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::AbstractFirm, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    continuationgrid = Grid(exantegrid, pricespace)
    innovationspace, signalweights = signal.space

    V = linear_interp(continuationgrid.nodes, firmvalue.continuation.V; extrap = constextrap)

    indices = CartesianIndices(firmvalue.exante)
    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        emissions = emissionsspace[i]
        z = reputationspace[j]
        p = logistic(z)
        τ = welfare.P[i, j]
        EV = zero(T)

        for (k, ξ) in enumerate(innovationspace)
            qⁿᶜ = realisedprice(ξ, τ, signal)
            qᶜ = realisedprice(ξ, τᶜ, signal)

            EV += signalweights[k] * (
                (1 - p) * evaluatefirmcontinuationvalue(V, emissions, z, qⁿᶜ, continuationgrid, τᶜ, firm, signal) +
                p * evaluatefirmcontinuationvalue(V, emissions, z, qᶜ, continuationgrid, τᶜ, firm, signal)
            )
        end

        firmvalue.exante[i, j] = EV
    end

    return firmvalue
end

function updateexanteemissions!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::AbstractFirm, signal::Signal, i) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    continuationgrid = Grid(exantegrid, pricespace)
    innovationspace, signalweights = signal.space

    V = linear_interp(continuationgrid.nodes, firmvalue.continuation.V; extrap = constextrap)
    emissions = emissionsspace[i]

    @inbounds Threads.@threads for j in eachindex(reputationspace)
        z = reputationspace[j]
        p = logistic(z)
        τ = welfare.P[i, j]
        EV = zero(T)

        for (k, ξ) in enumerate(innovationspace)
            qⁿᶜ = realisedprice(ξ, τ, signal)
            qᶜ = realisedprice(ξ, τᶜ, signal)

            EV += signalweights[k] * (
                (1 - p) * evaluatefirmcontinuationvalue(V, emissions, z, qⁿᶜ, continuationgrid, τᶜ, firm, signal) +
                p * evaluatefirmcontinuationvalue(V, emissions, z, qᶜ, continuationgrid, τᶜ, firm, signal)
            )
        end

        firmvalue.exante[i, j] = EV
    end

    return firmvalue
end

function setfirmboundaries!(firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::AbstractFirm, signal::Signal) where {T, FV <: FirmValue{T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, emissions) in enumerate(emissionsspace), (j, z) in enumerate(reputationspace)
        ω = (z - zmin) / (zmax - zmin)

        for (k, q) in enumerate(pricespace)
            v = v̲(emissions, q, firm) * (1 - ω) + v̄(emissions, q, τᶜ, firm, signal) * ω
            φ = φ̲(emissions, q, firm, signal) * (1 - ω) + φ̄(emissions, τᶜ, firm, signal) * ω

            firmvalue.continuation.V[i, j, k] = v
            firmvalue.continuation.P[i, j, k] = φ
        end

        firmvalue.exante[i, j] = m̄(emissions, τᶜ, firm, signal) * ω
    end

    return firmvalue
end

function setgovernmentboundaries!(welfare::TW, τᶜ, exantegrid::G, firm::AbstractFirm, government::Government, signal::Signal) where {T, TW <: ValueFunction{2, T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, emissions) in enumerate(emissionsspace), (j, z) in enumerate(reputationspace)
        ω = (z - zmin) / (zmax - zmin)

        welfare.V[i, j] = w̲(emissions, firm, government) * (1 - ω) + w̄(emissions, τᶜ, firm, government, signal) * ω
        welfare.P[i, j] = τ̲(emissions, firm, government) * (1 - ω) + τ̄(τᶜ, firm, government) * ω
    end

    return welfare
end

@inline function firmobjective(φ, emissions, z′, M::LI, exantegrid::G, τᶜ, firm::AbstractFirm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    emissions′ = f(φ, emissions, firm)
    exante = evaluatefirmexantevalue(M, emissions′, z′, exantegrid, τᶜ, firm, signal)

    return c(φ, firm) + firm.β * exante
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::AbstractFirm, signal::Signal; φlims = (0., 1.)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    M = linear_interp(exantegrid.nodes, firmvalue.exante; extrap = constextrap)

    indices = CartesianIndices(nextfirmvalue.continuation.V)
    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        emissions = emissionsspace[i]
        z = reputationspace[j]
        q = pricespace[k]
        τ = welfare.P[i, j]
        z′ = z + ℓ(q, τ, τᶜ, signal)
        φmax = min(φlims[2], investmentupper(emissions, firm))

        if φmax <= φlims[1]
            φ = φlims[1]
            value = firmobjective(φ, emissions, z′, M, exantegrid, τᶜ, firm, signal)
        else
            res = Optim.optimize(φ -> firmobjective(φ, emissions, z′, M, exantegrid, τᶜ, firm, signal), φlims[1], φmax, brent)
            φ = Optim.minimizer(res)
            value = Optim.minimum(res)
        end

        nextfirmvalue.continuation.V[i, j, k] = emissions * q + value
        nextfirmvalue.continuation.P[i, j, k] = φ
    end

    updateexantevalue!(nextfirmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal)

    return nextfirmvalue
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::FirmPermanentInvestment, signal::Signal; φlims = (0., 1.)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    emissionsspace, reputationspace = exantegrid.nodes
    copyto!(nextfirmvalue, firmvalue)
    emissionslower = sqrt(eps(T)) * max(one(T), firm.e₀)

    # Since e′ ≤ e, use already updated lower-emissions rows in the interpolation.
    @inbounds for (i, emissions) in enumerate(emissionsspace)
        if emissions <= emissionslower
            for j in eachindex(reputationspace)
                nextfirmvalue.exante[i, j] = zero(T)

                for k in eachindex(pricespace)
                    nextfirmvalue.continuation.V[i, j, k] = zero(T)
                    nextfirmvalue.continuation.P[i, j, k] = zero(T)
                end
            end

            continue
        end

        M = linear_interp(exantegrid.nodes, nextfirmvalue.exante; extrap = constextrap)

        Threads.@threads for idx in CartesianIndices((eachindex(reputationspace), eachindex(pricespace)))
            j, k = idx.I
            z = reputationspace[j]
            q = pricespace[k]
            τ = welfare.P[i, j]
            z′ = z + ℓ(q, τ, τᶜ, signal)
            φmax = min(φlims[2], investmentupper(emissions, firm))

            if φmax <= φlims[1]
                φ = φlims[1]
                value = firmobjective(φ, emissions, z′, M, exantegrid, τᶜ, firm, signal)
            else
                res = Optim.optimize(φ -> firmobjective(φ, emissions, z′, M, exantegrid, τᶜ, firm, signal), φlims[1], φmax, brent)
                φ = Optim.minimizer(res)
                value = Optim.minimum(res)
            end

            nextfirmvalue.continuation.V[i, j, k] = emissions * q + value
            nextfirmvalue.continuation.P[i, j, k] = φ
        end

        updateexanteemissions!(nextfirmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal, i)
    end

    return nextfirmvalue
end

function governmentobjective(τ, τᶜ, emissions, z, Φ::LIΦ, W::LIW, continuationgrid::GE, exantegrid::GA, firm::AbstractFirm, government::Government, signal::Signal) where {LIΦ <: FastInterpolations.AbstractInterpolant, LIW <: FastInterpolations.AbstractInterpolant, GE <: Grid{3}, GA <: Grid{2}}
    innovationspace, signalweights = signal.space

    EV = zero(τ)
    @inbounds for (k, ξ) in enumerate(innovationspace)
        q = realisedprice(ξ, τ, signal)
        z′ = z + ℓ(q, τ, τᶜ, signal)
        φ = evaluatefirmpolicy(Φ, emissions, z, q, continuationgrid, τᶜ, firm, signal)
        emissions′ = f(φ, emissions, firm)

        EV += signalweights[k] * (c(φ, firm) + government.β * evaluatewelfarevalue(W, emissions′, z′, exantegrid, τᶜ, firm, government, signal))
    end

    EV
end
