function applyfirmreputationboundaries!(firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    jmin = firstindex(reputationspace)
    jmax = lastindex(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace)
        firmvalue.exante[i, jmin] = zero(T)
        firmvalue.exante[i, jmax] = ψ̄(a, τᶜ, firm, signal)

        for (k, q) in enumerate(pricespace)
            firmvalue.continuation.V[i, jmin, k] = v̲(a, q, firm)
            firmvalue.continuation.P[i, jmin, k] = φ̲(a, q, firm, signal)
            firmvalue.continuation.V[i, jmax, k] = v̄(a, q, τᶜ, firm, signal)
            firmvalue.continuation.P[i, jmax, k] = φ̄(a, τᶜ, firm, signal)
        end
    end

    return firmvalue
end

function applygovernmentreputationboundaries!(welfare::TW, τᶜ, exantegrid::G, firm::Firm, government::Government, signal::Signal) where {T, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    jmin = firstindex(reputationspace)
    jmax = lastindex(reputationspace)

    @inbounds for (i, a) in enumerate(abatementspace)
        welfare.V[i, jmin] = w̲(a, firm, government)
        welfare.P[i, jmin] = τ̲(a, firm, government)
        welfare.V[i, jmax] = w̄(a, τᶜ, firm, government, signal)
        welfare.P[i, jmax] = τ̄(τᶜ, firm, government)
    end

    return welfare
end

function applyreputationboundaries!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    applyfirmreputationboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, signal)
    applygovernmentreputationboundaries!(welfare, τᶜ, exantegrid, firm, government, signal)

    return firmvalue, welfare
end

function selecteddeviation(τprev, devvalue, devpolicy, objective, τlims, τᶜ, policyopttol)
    lowvalue = objective(τlims[1])

    if lowvalue <= devvalue + policyopttol
        return lowvalue, τlims[1]
    end

    τprev = clamp(τprev, τlims[1], τlims[2])
    prevvalue = objective(τprev)

    if abs(τprev - τᶜ) > policyopttol && prevvalue <= devvalue + policyopttol
        return prevvalue, τprev
    end

    return devvalue, devpolicy
end

function optimizedeviation(objective, τlims, τᶜ, taxseparation)
    τmin, τmax = τlims
    taxseparation = max(zero(τᶜ), taxseparation)
    bestvalue = oftype(τᶜ, Inf)
    bestpolicy = τᶜ
    foundcandidate = false

    leftmax = min(τᶜ - taxseparation, τmax)
    if τmin <= leftmax
        if τmin == leftmax
            value = objective(τmin)
            policy = τmin
        else
            res = optimize(objective, τmin, leftmax, brent)
            value = Optim.minimum(res)
            policy = Optim.minimizer(res)
        end

        bestvalue = value
        bestpolicy = policy
        foundcandidate = true
    end

    rightmin = max(τᶜ + taxseparation, τmin)
    if rightmin <= τmax
        if rightmin == τmax
            value = objective(rightmin)
            policy = rightmin
        else
            res = optimize(objective, rightmin, τmax, brent)
            value = Optim.minimum(res)
            policy = Optim.minimizer(res)
        end

        if !foundcandidate || value < bestvalue
            bestvalue = value
            bestpolicy = policy
            foundcandidate = true
        end
    end

    if !foundcandidate
        return objective(τᶜ), τᶜ
    end

    return bestvalue, bestpolicy
end

function shouldmimic(gap, wasmimic, mimictol, mimicband)
    if wasmimic
        return gap <= max(mimictol, mimicband)
    else
        return gap <= mimictol
    end
end

function interiormimicshare(mimicmask, exantegrid)
    _, reputationspace = exantegrid.nodes
    interior = (firstindex(reputationspace) + 1):(lastindex(reputationspace) - 1)
    interiormask = view(mimicmask, :, interior)

    return count(interiormask) / length(interiormask)
end

function interiorstats(A, exantegrid)
    _, reputationspace = exantegrid.nodes
    interior = (firstindex(reputationspace) + 1):(lastindex(reputationspace) - 1)
    interiorview = view(A, :, interior)

    return minimum(interiorview), sum(interiorview) / length(interiorview), maximum(interiorview)
end

function governmentmimickingstep!(nextwelfare::TW, welfare::TW, firmvalue::FV, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; τlims = (0., 2τᶜ), mimictol = 1e-8, mimicband = 0., mimicmask = nothing, policyopttol = 1e-8, taxseparation = 0., mimicgap = nothing, deviationpolicy = nothing) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, reputationspace = exantegrid.nodes
    continuationgrid = Grid(exantegrid, pricespace)
    jmin = firstindex(reputationspace)
    jmax = lastindex(reputationspace)

    W = linear_interp(exantegrid.nodes, welfare.V; extrap = constextrap)
    Φ = linear_interp(continuationgrid.nodes, firmvalue.continuation.P; extrap = constextrap)

    indices = CartesianIndices(nextwelfare.V)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = reputationspace[j]
        damages = d(e(a, firm), government)

        if j == jmin
            nextwelfare.V[i, j] = w̲(a, firm, government)
            nextwelfare.P[i, j] = τ̲(a, firm, government)
            !isnothing(mimicmask) && (mimicmask[i, j] = false)
            !isnothing(mimicgap) && (mimicgap[i, j] = zero(T))
            !isnothing(deviationpolicy) && (deviationpolicy[i, j] = τlims[1])
            continue
        elseif j == jmax
            nextwelfare.V[i, j] = w̄(a, τᶜ, firm, government, signal)
            nextwelfare.P[i, j] = τ̄(τᶜ, firm, government)
            !isnothing(mimicmask) && (mimicmask[i, j] = true)
            !isnothing(mimicgap) && (mimicgap[i, j] = zero(T))
            !isnothing(deviationpolicy) && (deviationpolicy[i, j] = τᶜ)
            continue
        end

        objective = τ -> governmentobjective(τ, τᶜ, a, z, Φ, W, continuationgrid, exantegrid, firm, government, signal) + damages
        mimicvalue = objective(τᶜ)
        devvalue, devpolicy = optimizedeviation(objective, τlims, τᶜ, taxseparation)
        gap = mimicvalue - devvalue
        !isnothing(mimicgap) && (mimicgap[i, j] = gap)
        !isnothing(deviationpolicy) && (deviationpolicy[i, j] = devpolicy)

        mimic = if isnothing(mimicmask)
            gap <= mimictol
        else
            shouldmimic(gap, mimicmask[i, j], mimictol, mimicband)
        end

        if mimic
            nextwelfare.V[i, j] = mimicvalue
            nextwelfare.P[i, j] = τᶜ
        else
            devvalue, devpolicy = selecteddeviation(welfare.P[i, j], devvalue, devpolicy, objective, τlims, τᶜ, policyopttol)
            nextwelfare.V[i, j] = devvalue
            nextwelfare.P[i, j] = devpolicy
        end

        if !isnothing(mimicmask)
            mimicmask[i, j] = mimic
        end
    end

    return nextwelfare
end

function steadypolicystep!(nextfirmvalue::FV, nextwelfare::TW, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; φlims = (0., 1.), τlims = (0., 2τᶜ), mimictol = 1e-8, mimicband = 0., mimicmask = nothing, policyopttol = 1e-8, taxseparation = 0., mimicgap = nothing, deviationpolicy = nothing) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal; φlims)
    governmentmimickingstep!(nextwelfare, welfare, nextfirmvalue, τᶜ, exantegrid, pricespace, firm, government, signal; τlims, mimictol, mimicband, mimicmask, policyopttol, taxseparation, mimicgap, deviationpolicy)
    applyreputationboundaries!(nextfirmvalue, nextwelfare, τᶜ, exantegrid, pricespace, firm, government, signal)

    return nextfirmvalue, nextwelfare
end

function stationaryupdateerrors(nextfirmvalue::FV, nextwelfare::TW, firmvalue::FV, welfare::TW, exantegrid::G, valtol, poltol) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    _, reputationspace = exantegrid.nodes
    interior = (firstindex(reputationspace) + 1):(lastindex(reputationspace) - 1)

    εᶠᵛ = max(
        maximum(abs, view(nextfirmvalue.exante .- firmvalue.exante, :, interior)),
        maximum(abs, view(nextfirmvalue.continuation.V .- firmvalue.continuation.V, :, interior, :)),
    )
    εᶠₚ = maximum(abs, view(nextfirmvalue.continuation.P .- firmvalue.continuation.P, :, interior, :))
    εʷᵛ = maximum(abs, view(nextwelfare.V .- welfare.V, :, interior))
    εʷₚ = maximum(abs, view(nextwelfare.P .- welfare.P, :, interior))

    ε = normalizederror(max(εᶠᵛ, εʷᵛ), εᶠₚ, valtol, poltol)

    return εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε
end

function relaxstationaryupdate!(firmvalue::FV, welfare::TW, nextfirmvalue::FV, nextwelfare::TW, τᶜ, relax) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    @. firmvalue.exante = relax * nextfirmvalue.exante + (1 - relax) * firmvalue.exante
    @. firmvalue.continuation.V = relax * nextfirmvalue.continuation.V + (1 - relax) * firmvalue.continuation.V
    @. firmvalue.continuation.P = relax * nextfirmvalue.continuation.P + (1 - relax) * firmvalue.continuation.P
    @. welfare.V = relax * nextwelfare.V + (1 - relax) * welfare.V
    copyto!(welfare.P, nextwelfare.P)

    @. firmvalue.continuation.P = clamp(firmvalue.continuation.P, 0, 1)
    @. welfare.P = clamp(welfare.P, 0, 2τᶜ)

    return firmvalue, welfare
end

function steadypolicies!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; initialize = true, maxiter = 500, relax = 0.1, valtol = 1e-8, poltol = 1e-4, φlims = (0., 1.), τlims = (0., 2τᶜ), mimictol = 1e-8, mimicband = 10mimictol, policyopttol = 1e-8, taxseparation = 0., mimicmask = nothing, verbose = 0) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    if initialize
        setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, signal)
        setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, signal)
        applyreputationboundaries!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal)
    end

    if isnothing(mimicmask)
        mimicmask = abs.(welfare.P .- τᶜ) .<= sqrt(eps(T)) * max(one(T), abs(τᶜ))
    end

    nextfirmvalue = similar(firmvalue)
    nextwelfare = similar(welfare)
    mimicgap = verbose > 1 ? similar(welfare.V) : nothing
    deviationpolicy = verbose > 1 ? similar(welfare.P) : nothing
    relax = T(relax)

    for iter in 1:maxiter
        steadypolicystep!(nextfirmvalue, nextwelfare, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; φlims, τlims, mimictol, mimicband, mimicmask, policyopttol, taxseparation, mimicgap, deviationpolicy)
        εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = stationaryupdateerrors(nextfirmvalue, nextwelfare, firmvalue, welfare, exantegrid, valtol, poltol)

        if verbose > 0
            mimicshare = interiormimicshare(mimicmask, exantegrid)
            @printf "Stationary policy iteration %d: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e, normalized = %.2e, interior mimic share = %.2f\n" iter εᶠᵛ εᶠₚ εʷᵛ εʷₚ ε mimicshare
        end

        if verbose > 1
            gapmin, gapmean, gapmax = interiorstats(mimicgap, exantegrid)
            devtaxmin, _, devtaxmax = interiorstats(deviationpolicy, exantegrid)
            @printf "  separated deviation gain = [%.2e, %.2e, %.2e], separated tax = [%.2e, %.2e]\n" gapmin gapmean gapmax devtaxmin devtaxmax
        end

        if ε < 1
            relaxstationaryupdate!(firmvalue, welfare, nextfirmvalue, nextwelfare, τᶜ, one(T))
            return iter, firmvalue, welfare
        end

        relaxstationaryupdate!(firmvalue, welfare, nextfirmvalue, nextwelfare, τᶜ, relax)
        applyreputationboundaries!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal)
    end

    if verbose > 0
        @warn @sprintf "Stationary policy iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmvalue, welfare
end

function homotopysteadypolicies!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; σpath::TP = [signal.σ], maxiter = 500, relax = 0.1, valtol = 1e-8, poltol = 1e-4, φlims = (0., 1.), τlims = (0., 2τᶜ), mimictol = 1e-8, mimicband = 10mimictol, policyopttol = 1e-8, taxseparation = 0., verbose = 0) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}, TP <: AbstractVector{T}}
    initialsignal = Signal(signal.μ, σpath[1], signal.space)
    setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, initialsignal)
    setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, initialsignal)
    applyreputationboundaries!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, initialsignal)

    mimicmask = abs.(welfare.P .- τᶜ) .<= sqrt(eps(T)) * max(one(T), abs(τᶜ))
    iterations = Vector{Int}(undef, length(σpath))

    for (i, σᵢ) in enumerate(σpath)
        if verbose > 0
            @printf "\nStationary policy homotopy %d/%d, σ = %.4f\n" i length(σpath) σᵢ
        end

        signalᵢ = Signal(signal.μ, σᵢ, signal.space)
        iter, _, _ = steadypolicies!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signalᵢ; initialize = false, maxiter, relax, valtol, poltol, φlims, τlims, mimictol, mimicband, policyopttol, taxseparation, mimicmask, verbose)
        iterations[i] = iter
    end

    return iterations, firmvalue, welfare
end
