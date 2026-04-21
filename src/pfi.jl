const sqrtπ = √π

const constextrap = ConstExtrap()
const brent = Brent()

function firmobjective(φ, a, z′, Ψ, firm::Firm)
    a′ = f(φ, a, firm)

    return c(φ, firm) + firm.β * Ψ((a′, z′))
end

"Expected firm continuation value before the realization of q."
function updatefirmcontinuation!(firmvalue::FV, welfare::TW, τᶜ, grid::G, qspace, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes
    innovationspace, signalweights = signal.space
    indices = firmindices(grid)

    V = linear_interp((abatementspace, logitspace, qspace), firmvalue.expost.V; extrap = constextrap)
    Φ = linear_interp((abatementspace, logitspace, qspace), firmvalue.expost.P; extrap = constextrap)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = logitspace[j]
        p = logistic(z)
        τ = welfare.P[i, j]
        EV = zero(T)
        Eφ = zero(T)

        for (k, ξ) in enumerate(innovationspace)
            qⁿᶜ = signalprice(ξ, τ, signal)
            qᶜ = signalprice(ξ, τᶜ, signal)

            EV += signalweights[k] * ((1 - p) * V((a, z, qⁿᶜ)) + p * V((a, z, qᶜ)))
            Eφ += signalweights[k] * ((1 - p) * Φ((a, z, qⁿᶜ)) + p * Φ((a, z, qᶜ)))
        end

        firmvalue.continuation.V[i, j] = EV / sqrtπ
        firmvalue.continuation.P[i, j] = Eφ / sqrtπ
    end

    return firmvalue
end

function setfirmboundaries!(firmvalue::FV, τᶜ, grid::G, qspace, firm::Firm, signal::Signal) where {T, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes
    zmin, zmax = extrema(logitspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(logitspace)
        ω = (z - zmin) / (zmax - zmin)
        ψ = ψ̲(a, firm, signal) * (1 - ω) + ψ̄(a, τᶜ, firm, signal) * ω
        φ = φ̲(a, zero(T), firm, signal) * (1 - ω) + φ̄(τᶜ, firm, signal) * ω

        firmvalue.continuation.V[i, j] = ψ
        firmvalue.continuation.P[i, j] = φ
    end

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(logitspace), (k, q) in enumerate(qspace)
        ω = (z - zmin) / (zmax - zmin)
        v = v̲(a, q, firm) * (1 - ω) + v̄(a, q, τᶜ, firm, signal) * ω
        φ = φ̲(a, q, firm, signal) * (1 - ω) + φ̄(τᶜ, firm, signal) * ω

        firmvalue.expost.V[i, j, k] = v
        firmvalue.expost.P[i, j, k] = φ
    end

    return firmvalue
end

function setgovernmentboundaries!(welfare::TW, τᶜ, grid::G, firm::Firm, government::Government, signal::Signal) where {T, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes
    zmin, zmax = extrema(logitspace)
    
    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(logitspace)
        ω = (z - zmin) / (zmax - zmin)

        welfare.V[i, j] = w̲(a, firm, government) * (1 - ω) + w̄(a, τᶜ, firm, government, signal) * ω
        welfare.P[i, j] = τ̲(a, firm, government) * (1 - ω) + τ̄(τᶜ, firm, government) * ω
    end

    return welfare
end

function logitinteriorindices(grid::G) where G <: Grid{2} 
    abatementspace, logitspace = grid.nodes
    interiorlogit = (firstindex(logitspace) + 1):(lastindex(logitspace) - 1)

    nodes = (axes(abatementspace, 1), interiorlogit)
    
    return CartesianIndices(nodes)
end

function logitinteriorindices(grid::G, qspace::AbstractVector) where G <: Grid{2} 
    abatementspace, logitspace = grid.nodes
    interiorlogit = (firstindex(logitspace) + 1):(lastindex(logitspace) - 1)

    nodes = (axes(abatementspace, 1), interiorlogit, axes(qspace, 1))
    
    return CartesianIndices(nodes)
end

function firmindices(grid::G) where G <: Grid{2}
    abatementspace, logitspace = grid.nodes
    logitindices = (firstindex(logitspace) + 1):lastindex(logitspace)

    nodes = (axes(abatementspace, 1), logitindices)

    return CartesianIndices(nodes)
end

function firmindices(grid::G, qspace::AbstractVector) where G <: Grid{2}
    abatementspace, logitspace = grid.nodes
    logitindices = (firstindex(logitspace) + 1):lastindex(logitspace)

    nodes = (axes(abatementspace, 1), logitindices, axes(qspace, 1))

    return CartesianIndices(nodes)
end

function governmentindices(grid::G) where G <: Grid{2}
    abatementspace, logitspace = grid.nodes
    logitindices = (firstindex(logitspace) + 1):lastindex(logitspace)

    nodes = (axes(abatementspace, 1), logitindices)

    return CartesianIndices(nodes)
end

function firmstep!(nextfirmvalue::FV, firmvalue::FV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, signal::Signal; φmax = one(T)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes

    indices = firmindices(grid, qspace)
    Ψ = linear_interp(grid.nodes, firmvalue.continuation.V; extrap = constextrap)

    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        a = abatementspace[i]
        z = logitspace[j]    
        q = qspace[k]
        τ = welfare.P[i, j]
        z′ = z + logitdrift(q, τ, τᶜ, signal)

        res = optimize(φ -> firmobjective(φ, a, z′, Ψ, firm), zero(T), φmax, brent)

        nextfirmvalue.expost.V[i, j, k] = e(a, firm) * q + Optim.minimum(res)
        nextfirmvalue.expost.P[i, j, k] = Optim.minimizer(res)
    end

    updatefirmcontinuation!(nextfirmvalue, welfare, τᶜ, grid, qspace, signal)

    return nextfirmvalue
end

function solvefirm!(firmvalue::FV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, signal::Signal; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    
    nextfirmvalue = copy(firmvalue)
    for iter in 1:maxiter
        firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, grid, qspace, firm, signal; iterkwargs...)
        
        εᵥ = max(
            maximum(abs, nextfirmvalue.continuation.V .- firmvalue.continuation.V),
            maximum(abs, nextfirmvalue.expost.V .- firmvalue.expost.V),
        )
        εₚ = maximum(abs, nextfirmvalue.expost.P .- firmvalue.expost.P)
        
        copyto!(firmvalue,  nextfirmvalue)
        
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

    for (k, ξ) in enumerate(innovationspace)
        q = signalprice(ξ, τ, signal)
        φ = Φ((a, z, q))
        a′ = f(φ, a, firm)
        z′ = z + logitdrift(q, τ, τᶜ, signal)

        EV += signalweights[k] * (c(φ, firm) + government.β * W((a′, z′)))
    end

    EV / sqrtπ
end

function governmentstep!(nextwelfare::TW, welfare::TW, firmvalue::FV, τᶜ, grid::G, qspace, firm::Firm, government::Government, signal::Signal; τmax = T(100.0), τgridpoints = 101) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes

    W = linear_interp(grid.nodes, welfare.V; extrap = ConstExtrap())
    Φ = linear_interp((abatementspace, logitspace, qspace), firmvalue.expost.P; extrap = ConstExtrap())

    τgrid = range(zero(T), τmax, τgridpoints)
    indices = governmentindices(grid)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = logitspace[j]
        
        τopt = τᶜ # Begins by checking mimicking
        wopt = governmentobjective(τopt, τᶜ, a, z, Φ, W, firm, government, signal)

        @inbounds for τᵢ in τgrid
            wᵢ = governmentobjective(τᵢ, τᶜ, a, z, Φ, W, firm, government, signal)
            
            if wᵢ < wopt
                τopt = τᵢ
                wopt = wᵢ
            end
        end

        nextwelfare.V[i, j] = wopt + d(e(a, firm), government)
        nextwelfare.P[i, j] = τopt
    end

    return nextwelfare
end

function solvegovernment!(welfare::TW, firmvalue::FV, τᶜ, grid::G, qspace, firm::Firm, signal::Signal, government::Government; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TW <: ValueFunction{2, T}, FV <: FirmValue{T}, G <: Grid{2}}

    nextwelfare = copy(welfare)
    for iter in 1:maxiter
        governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, grid, qspace, firm, government, signal; iterkwargs...)

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

function nestedpfi!(firmvalue::FV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, government::Government, signal::Signal; maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0, firmparams = Dict{Symbol, T}(), welfareparams = Dict{Symbol, T}()) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    
    setfirmboundaries!(firmvalue, τᶜ, grid, qspace, firm, signal)
    setgovernmentboundaries!(welfare, τᶜ, grid, firm, government, signal)
    
    oldwelfare = similar(welfare)
    for iter in 1:maxiter
        copyto!(oldwelfare, welfare)

        firmiter, _ = solvefirm!(firmvalue, welfare, τᶜ, grid, qspace, firm, signal; verbose, firmparams...)
        goviter, _ = solvegovernment!(welfare, firmvalue, τᶜ, grid, qspace, firm, signal, government; verbose, welfareparams...)

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
