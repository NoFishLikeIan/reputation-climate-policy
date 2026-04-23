const brent = Optim.Brent()
const constextrap = ConstExtrap()

function relaxupdate!(valuefunction::TV, oldvaluefunction::TV, relax) where TV <: ValueFunction
    @. valuefunction.V = relax * valuefunction.V + (1 - relax) * oldvaluefunction.V
    @. valuefunction.P = relax * valuefunction.P + (1 - relax) * oldvaluefunction.P

    return valuefunction
end

function relaxupdate!(firmvalue::FV, oldfirmvalue::FV, relax) where FV <: FirmValue
    relaxupdate!(firmvalue.continuation, oldfirmvalue.continuation, relax)
    relaxupdate!(firmvalue.expost, oldfirmvalue.expost, relax)

    return firmvalue
end

"Expected firm continuation value before the realization of q."
function updatecontinuationvalue!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    expostgrid = Grid(exantegrid, pricespace)
    innovationspace, signalweights = signal.space

    V = linear_interp(expostgrid.nodes, firmvalue.expost.V; extrap = constextrap)

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
                (1 - p) * evaluatefirmexpostvalue(V, a, z, qⁿᶜ, expostgrid, τᶜ, firm, signal) +
                p * evaluatefirmexpostvalue(V, a, z, qᶜ, expostgrid, τᶜ, firm, signal)
            )
        end

        firmvalue.continuation.V[i, j] = EV
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

            firmvalue.expost.V[i, j, k] = v
            firmvalue.expost.P[i, j, k] = φ
        end

        firmvalue.continuation.V[i, j] = ψ̄(a, τᶜ, firm, signal) * ω
        firmvalue.continuation.P[i, j] = T(NaN)
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
    continuation = evaluatefirmcontinuation(Ψ, a′, z′, exantegrid, τᶜ, firm, signal)

    return c(φ, firm) + firm.β * continuation
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal; φlims = (0., 1.)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    Ψ = linear_interp(exantegrid.nodes, firmvalue.continuation.V; extrap = constextrap)

    indices = CartesianIndices(nextfirmvalue.expost.V)
    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        a = abatementspace[i]
        z = reputationspace[j]
        q = pricespace[k]
        τ = welfare.P[i, j]
        z′ = z + ℓ(q, τ, τᶜ, signal)

        res = Optim.optimize(φ -> firmobjective(φ, a, z′, Ψ, exantegrid, τᶜ, firm, signal), φlims[1], φlims[2], brent)

        nextfirmvalue.expost.V[i, j, k] = e(a, firm) * q + Optim.minimum(res)
        nextfirmvalue.expost.P[i, j, k] = Optim.minimizer(res)
    end

    updatecontinuationvalue!(nextfirmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal)

    return nextfirmvalue
end

function solvefirm!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal; maxiter = 500, valtol = 1e-8, poltol = 1e-4, improvetol = 1e-3, worsetol = 1e-3, maxstall = 5, verbose = 0, iterkwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    nextfirmvalue = copy(firmvalue)
    prevε = T(Inf)
    stalliter = 0

    for iter in 1:maxiter
        firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal; iterkwargs...)

        εᵥ = maximum(abs, nextfirmvalue.expost.V .- firmvalue.expost.V)
        εₚ = maximum(abs, nextfirmvalue.expost.P .- firmvalue.expost.P)
        ε = normalizederror(εᵥ, εₚ, valtol, poltol)

        if verbose > 1
            @printf "Firm iteration %d, value error = %.2e, policy error %.2e\r" iter εᵥ εₚ
        end

        if ε < one(ε)
            copyto!(firmvalue, nextfirmvalue)
            return iter, firmvalue
        elseif errorincreased(ε, prevε, worsetol)
            if verbose > 1
                @warn @sprintf "Firm iteration stopped after %d iterations because the normalized error increased\n" iter
            end

            return iter, firmvalue
        end

        copyto!(firmvalue, nextfirmvalue)

        if insufficientimprovement(ε, prevε, improvetol)
            stalliter += 1

            if stalliter >= maxstall
                if verbose > 1
                    @warn @sprintf "Firm iteration stopped after %d iterations because the normalized error improved too slowly\n" iter
                end

                return iter, firmvalue
            end
        else
            stalliter = 0
        end

        prevε = ε
    end

    if verbose > 1
        @warn @sprintf "Firm iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmvalue
end

function governmentobjective(τ, τᶜ, a, z, Φ::LIΦ, W::LIW, expostgrid::GE, exantegrid::GA, firm::Firm, government::Government, signal::Signal) where {LIΦ <: FastInterpolations.AbstractInterpolant, LIW <: FastInterpolations.AbstractInterpolant, GE <: Grid{3}, GA <: Grid{2}}
    innovationspace, signalweights = signal.space

    EV = zero(τ)
    @inbounds for (k, ξ) in enumerate(innovationspace)
        q = realisedprice(ξ, τ, signal)
        z′ = z + ℓ(q, τ, τᶜ, signal)
        φ = evaluatefirmpolicy(Φ, a, z, q, expostgrid, τᶜ, firm, signal)
        a′ = f(φ, a, firm)

        EV += signalweights[k] * (c(φ, firm) + government.β * evaluatewelfarevalue(W, a′, z′, exantegrid, τᶜ, firm, government, signal))
    end

    EV
end

function governmentstep!(nextwelfare::TW, welfare::TW, firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; τlims = (0., 2τᶜ)) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    expostgrid = Grid(exantegrid, pricespace)

    W = linear_interp(exantegrid.nodes, welfare.V; extrap = constextrap)
    Φ = linear_interp(expostgrid.nodes, firmvalue.expost.P; extrap = constextrap)

    indices = CartesianIndices(nextwelfare.V)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = reputationspace[j]

        res = optimize(τ -> governmentobjective(τ, τᶜ, a, z, Φ, W, expostgrid, exantegrid, firm, government, signal), τlims[1], τlims[2], brent)

        nextwelfare.V[i, j] = Optim.minimum(res) + d(e(a, firm), government)
        nextwelfare.P[i, j] = Optim.minimizer(res)
    end

    return nextwelfare
end

function solvegovernment!(welfare::TW, firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal, government::Government; maxiter = 500, valtol = 1e-8, poltol = 1e-4, improvetol = 1e-3, worsetol = 1e-3, maxstall = 5, verbose = 0, iterkwargs...) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    nextwelfare = copy(welfare)
    prevε = T(Inf)
    stalliter = 0

    for iter in 1:maxiter
        governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, exantegrid, pricespace, firm, government, signal; iterkwargs...)

        εᵥ = maximum(abs, nextwelfare.V .- welfare.V)
        εₚ = maximum(abs, nextwelfare.P .- welfare.P)
        ε = normalizederror(εᵥ, εₚ, valtol, poltol)

        if verbose > 1
            @printf "Government iteration %d, value error = %.2e, policy error %.2e\r" iter εᵥ εₚ
        end

        if ε < one(ε)
            copyto!(welfare, nextwelfare)
            return iter, welfare
        elseif errorincreased(ε, prevε, worsetol)
            if verbose > 1
                @warn @sprintf "Government iteration stopped after %d iterations because the normalized error increased\n" iter
            end

            return iter, welfare
        end

        copyto!(welfare, nextwelfare)

        if insufficientimprovement(ε, prevε, improvetol)
            stalliter += 1

            if stalliter >= maxstall
                if verbose > 1
                    @warn @sprintf "Government iteration stopped after %d iterations because the normalized error improved too slowly\n" iter
                end

                return iter, welfare
            end
        else
            stalliter = 0
        end

        prevε = ε
    end

    if verbose > 1
        @warn @sprintf "Government iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, welfare
end

function nestedpfi!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; maxiter = 100, valtol = 1e-8, poltol = 1e-4, improvetol = 1e-3, worsetol = 1e-3, maxstall = 5, firmrelax = 0.25, welfarerelax = 0.25, verbose = 0, firmparams = Dict{Symbol, T}(), welfareparams = Dict{Symbol, T}()) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, signal)
    setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, signal)

    oldfirmvalue = similar(firmvalue)
    oldwelfare = similar(welfare)
    prevε = T(Inf)
    stalliter = 0

    for iter in 1:maxiter
        copyto!(oldfirmvalue, firmvalue)
        copyto!(oldwelfare, welfare)

        firmiter, _ = solvefirm!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal; verbose, firmparams...)
        goviter, _ = solvegovernment!(welfare, firmvalue, τᶜ, exantegrid, pricespace, firm, signal, government; verbose, welfareparams...)

        if firmrelax < 1
            relaxupdate!(firmvalue, oldfirmvalue, firmrelax)
        end

        if welfarerelax < 1
            relaxupdate!(welfare, oldwelfare, welfarerelax)
        end

        εᶠᵥ = max(maximum(abs, oldfirmvalue.continuation.V .- firmvalue.continuation.V), maximum(abs, oldfirmvalue.expost.V .- firmvalue.expost.V))
        εᶠₚ = maximum(abs, oldfirmvalue.expost.P .- firmvalue.expost.P)
        εʷᵥ = maximum(abs, oldwelfare.V .- welfare.V)
        εʷₚ = maximum(abs, oldwelfare.P .- welfare.P)
        εᵥ = max(εᶠᵥ, εʷᵥ)
        εₚ = max(εᶠₚ, εʷₚ)
        ε = normalizederror(εᵥ, εₚ, valtol, poltol)

        if verbose > 0
            @printf "\nNested iteration %d, firm iters = %d, gov iters = %d, firm value error = %.2e, firm policy error %.2e, welfare value error = %.2e, welfare policy error %.2e\n" iter firmiter goviter εᶠᵥ εᶠₚ εʷᵥ εʷₚ
        end

        if ε < one(ε)
            return iter, firmvalue, welfare
        elseif errorincreased(ε, prevε, worsetol)
            copyto!(firmvalue, oldfirmvalue)
            copyto!(welfare, oldwelfare)

            if verbose > 0
                @warn @sprintf "Nested policy iteration stopped after %d iterations because the normalized error increased\n" iter
            end

            return iter, firmvalue, welfare
        elseif insufficientimprovement(ε, prevε, improvetol)
            stalliter += 1

            if stalliter >= maxstall
                if verbose > 0
                    @warn @sprintf "Nested policy iteration stopped after %d iterations because the normalized error improved too slowly\n" iter
                end

                return iter, firmvalue, welfare
            end
        else
            stalliter = 0
        end

        prevε = ε
    end

    if verbose > 0
        @warn @sprintf "Nested policy iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmvalue, welfare
end
