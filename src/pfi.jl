const constextrap = ConstExtrap()
const extendextrap = ExtendExtrap()
const brent = Brent()

@inline clampnode(x, space) = clamp(x, first(space), last(space))

@inline function evaluatefirmexpostvalue(V, a, z, q, abatementspace, reputationspace, pricespace, τᶜ, firm::Firm, signal::Signal)
    aclamped = clampnode(a, abatementspace)
    qclamped = clampnode(q, pricespace)

    if z <= first(reputationspace)
        return v̲(aclamped, qclamped, firm)
    elseif z >= last(reputationspace)
        return v̄(aclamped, qclamped, τᶜ, firm, signal)
    end

    return V((aclamped, z, qclamped))
end

@inline function evaluatefirmpolicy(Φ, a, z, q, abatementspace, reputationspace, pricespace, τᶜ, firm::Firm, signal::Signal)
    aclamped = clampnode(a, abatementspace)
    qclamped = clampnode(q, pricespace)

    if z <= first(reputationspace)
        return φ̲(aclamped, qclamped, firm, signal)
    elseif z >= last(reputationspace)
        return φ̄(τᶜ, firm, signal)
    end

    return Φ((aclamped, z, qclamped))
end

@inline function evaluatefirmcontinuation(Ψ, a, z, abatementspace, reputationspace, τᶜ, firm::Firm, signal::Signal)
    aclamped = clampnode(a, abatementspace)

    if z <= first(reputationspace)
        return zero(aclamped)
    elseif z >= last(reputationspace)
        return ψ̄(aclamped, τᶜ, firm, signal)
    end

    return Ψ((aclamped, z))
end

@inline function evaluatewelfarevalue(W, a, z, abatementspace, reputationspace, τᶜ, firm::Firm, government::Government, signal::Signal)
    aclamped = clampnode(a, abatementspace)

    if z <= first(reputationspace)
        return w̲(aclamped, firm, government)
    elseif z >= last(reputationspace)
        return w̄(aclamped, τᶜ, firm, government, signal)
    end

    return W((aclamped, z))
end

"Expected firm continuation value before the realization of q."
function updatecontinuationvalue!(firmvalue::FV, welfare::TW, τᶜ, grid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = grid.nodes
    innovationspace, signalweights = signal.space

    V = linear_interp((abatementspace, reputationspace, pricespace), firmvalue.expost.V; extrap = constextrap)

    indices = CartesianIndices(firmvalue.continuation.V)
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
                (1 - p) * evaluatefirmexpostvalue(V, a, z, qⁿᶜ, abatementspace, reputationspace, pricespace, τᶜ, firm, signal) +
                p * evaluatefirmexpostvalue(V, a, z, qᶜ, abatementspace, reputationspace, pricespace, τᶜ, firm, signal)
            )
        end

        firmvalue.continuation.V[i, j] = EV
    end

    return firmvalue
end

function setfirmboundaries!(firmvalue::FV, τᶜ, grid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, G <: Grid{2}}
    signalweights = signal.space[2]
    abatementspace, reputationspace = grid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(reputationspace)
        EΨ = zero(T)
        ω = (z - zmin) / (zmax - zmin)

        for (k, q) in enumerate(pricespace)
            v = v̲(a, q, firm) * (1 - ω) + v̄(a, q, τᶜ, firm, signal) * ω
            φ = φ̲(a, q, firm, signal) * (1 - ω) + φ̄(τᶜ, firm, signal) * ω

            firmvalue.expost.V[i, j, k] = v
            firmvalue.expost.P[i, j, k] = φ

            EΨ += signalweights[k] * v
        end

        firmvalue.continuation.V[i, j] = EΨ
        firmvalue.continuation.P[i, j] = T(NaN)
    end

    return firmvalue
end

function setgovernmentboundaries!(welfare::TW, τᶜ, grid::G, firm::Firm, government::Government, signal::Signal) where {T, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = grid.nodes
    zmin, zmax = extrema(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(reputationspace)
        ω = (z - zmin) / (zmax - zmin)

        welfare.V[i, j] = w̲(a, firm, government) * (1 - ω) + w̄(a, τᶜ, firm, government, signal) * ω
        welfare.P[i, j] = τ̲(a, firm, government) * (1 - ω) + τ̄(τᶜ, firm, government) * ω
    end

    return welfare
end

function firmobjective(φ, a, z′, Ψ, firm::Firm)
    a′ = f(φ, a, firm)

    return c(φ, firm) + firm.β * Ψ(a′, z′)
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, grid::G, pricespace, firm::Firm, signal::Signal; φlims = (0., 1.)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = grid.nodes
    Ψinterp = linear_interp(grid.nodes, firmvalue.continuation.V; extrap = constextrap)
    Ψ = (a, z) -> evaluatefirmcontinuation(Ψinterp, a, z, abatementspace, reputationspace, τᶜ, firm, signal)

    indices = CartesianIndices(nextfirmvalue.expost.V)
    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        a = abatementspace[i]
        z = reputationspace[j]
        q = pricespace[k]
        τ = welfare.P[i, j]
        z′ = z + ℓ(q, τ, τᶜ, signal)

        res = Optim.optimize(φ -> firmobjective(φ, a, z′, Ψ, firm), φlims[1], φlims[2], brent)

        nextfirmvalue.expost.V[i, j, k] = e(a, firm) * q + Optim.minimum(res)
        nextfirmvalue.expost.P[i, j, k] = Optim.minimizer(res)
    end

    updatecontinuationvalue!(nextfirmvalue, welfare, τᶜ, grid, pricespace, firm, signal)

    return nextfirmvalue
end

function solvefirm!(firmvalue::FV, welfare::TW, τᶜ, grid::G, pricespace, firm::Firm, signal::Signal; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    nextfirmvalue = copy(firmvalue)
    for iter in 1:maxiter
        firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, grid, pricespace, firm, signal; iterkwargs...)

        εᵥ = maximum(abs, nextfirmvalue.expost.V .- firmvalue.expost.V)
        εₚ = maximum(abs, nextfirmvalue.expost.P .- firmvalue.expost.P)

        copyto!(firmvalue, nextfirmvalue)

        if verbose > 1
            @printf "Firm iteration %d, value error = %.2e, policy error %.2e\r" iter εᵥ εₚ
        end

        if (εᵥ < valtol && εₚ < poltol)
            return iter, firmvalue
        end
    end

    if verbose > 1
        @warn @sprintf "Firm iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmvalue
end

function governmentobjective(τ, τᶜ, a, z, Φ, W, firm, government, signal)
    innovationspace, signalweights = signal.space

    EV = zero(τ)
    @inbounds for (k, ξ) in enumerate(innovationspace)
        q = realisedprice(ξ, τ, signal)
        z′ = z + ℓ(q, τ, τᶜ, signal)
        φ = Φ(a, z, q)
        a′ = f(φ, a, firm)

        EV += signalweights[k] * (c(φ, firm) + government.β * W(a′, z′))
    end

    EV
end

function governmentstep!(nextwelfare::TW, welfare::TW, firmvalue::FV, τᶜ, grid::G, pricespace, firm::Firm, government::Government, signal::Signal; τlims = (0., 2τᶜ)) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, reputationspace = grid.nodes

    Winterp = linear_interp(grid.nodes, welfare.V; extrap = constextrap)
    Φinterp = linear_interp((abatementspace, reputationspace, pricespace), firmvalue.expost.P; extrap = constextrap)
    W = (a, z) -> evaluatewelfarevalue(Winterp, a, z, abatementspace, reputationspace, τᶜ, firm, government, signal)
    Φ = (a, z, q) -> evaluatefirmpolicy(Φinterp, a, z, q, abatementspace, reputationspace, pricespace, τᶜ, firm, signal)

    indices = CartesianIndices(nextwelfare.V)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = reputationspace[j]

        res = optimize(τ -> governmentobjective(τ, τᶜ, a, z, Φ, W, firm, government, signal), τlims[1], τlims[2], brent)

        nextwelfare.V[i, j] = Optim.minimum(res) + d(e(a, firm), government)
        nextwelfare.P[i, j] = Optim.minimizer(res)
    end

    return nextwelfare
end

function solvegovernment!(welfare::TW, firmvalue::FV, τᶜ, grid::G, pricespace, firm::Firm, signal::Signal, government::Government; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    nextwelfare = copy(welfare)
    for iter in 1:maxiter
        governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, grid, pricespace, firm, government, signal; iterkwargs...)

        εᵥ = maximum(abs, nextwelfare.V .- welfare.V)
        εₚ = maximum(abs, nextwelfare.P .- welfare.P)

        copyto!(welfare, nextwelfare)

        if verbose > 1
            @printf "Government iteration %d, value error = %.2e, policy error %.2e\r" iter εᵥ εₚ
        end

        if (εᵥ < valtol && εₚ < poltol)
            return iter, welfare
        end
    end

    if verbose > 1
        @warn @sprintf "Government iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, welfare
end

function nestedpfi!(firmvalue::FV, welfare::TW, τᶜ, grid::G, pricespace, firm::Firm, government::Government, signal::Signal; maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0, firmparams = Dict{Symbol, T}(), welfareparams = Dict{Symbol, T}()) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    setfirmboundaries!(firmvalue, τᶜ, grid, pricespace, firm, signal)
    setgovernmentboundaries!(welfare, τᶜ, grid, firm, government, signal)

    oldwelfare = similar(welfare)
    for iter in 1:maxiter
        copyto!(oldwelfare, welfare)

        firmiter, _ = solvefirm!(firmvalue, welfare, τᶜ, grid, pricespace, firm, signal; verbose, firmparams...)
        goviter, _ = solvegovernment!(welfare, firmvalue, τᶜ, grid, pricespace, firm, signal, government; verbose, welfareparams...)

        εᵥ = maximum(abs, oldwelfare.V .- welfare.V)
        εₚ = maximum(abs, oldwelfare.P .- welfare.P)

        if verbose > 0
            @printf "\nNested iteration %d, firm iters = %d, gov iters = %d, value error = %.2e, policy error %.2e\n" iter firmiter goviter εᵥ εₚ
        end

        if (εᵥ < valtol && εₚ < poltol)
            return iter, firmvalue, welfare
        end
    end

    if verbose > 0
        @warn @sprintf "Nested policy iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmvalue, welfare
end
