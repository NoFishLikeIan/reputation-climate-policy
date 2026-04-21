const sqrtπ = √π

const constextrap = ConstExtrap()
const brent = Brent()

"Expected firm value, with q as the ex-post third state."
function continuationvalue(valuefunction::TV, a′, z′, τ′, τᶜ, qspace, firm::Firm, signal::Signal, grid::G) where {T, TV <: ValueFunction{3, T}, G <: Grid{2}}
    innovationspace, signalweights = signal.space
    nodes = (grid.nodes[1], grid.nodes[2], qspace)
    
    p′ = logistic(z′)
    EV = zero(T)
    
    @inbounds for (k, ξ) in enumerate(innovationspace)
        qⁿᶜ = signalprice(ξ, τ′, signal)
        qᶜ = signalprice(ξ, τᶜ, signal)
        vⁿᶜ = linear_interp(nodes, valuefunction.V, (a′, z′, qⁿᶜ); extrap = constextrap)
        vᶜ  = linear_interp(nodes, valuefunction.V, (a′, z′, qᶜ); extrap = constextrap)
        EV  += signalweights[k] * ((1 - p′) * vⁿᶜ + p′ * vᶜ)
    end

    return EV / sqrtπ
end

function setfirmboundaries!(firmvalue::TV, τᶜ, grid::G, qspace, firm::Firm, signal::Signal) where {T, TV <: ValueFunction{3, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes
    zmin, zmax = extrema(logitspace)

    @inbounds for (i, a) in enumerate(abatementspace), (j, z) in enumerate(logitspace), (k, q) in enumerate(qspace)
        ω = (z - zmin) / (zmax - zmin)
        v = v̲(a, q, firm) * (1 - ω) + v̄(a, q, τᶜ, firm, signal) * ω
        φ = φ̲(a, q, firm, signal) * (1 - ω) + φ̄(τᶜ, firm, signal) * ω

        firmvalue.V[i, j, k] = v
        firmvalue.P[i, j, k] = φ
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

function firmobjective(φ, a, z′, Θ, τᶜ, firmvalue, qspace, grid, firm, signal)
    a′  = f(φ, a, firm)
    x′ = (a′, z′)
    τ′ = Θ(x′)
    return c(φ, firm) + firm.β * continuationvalue(firmvalue, a′, z′, τ′, τᶜ, qspace, firm, signal, grid)
end

function firmstep!(nextfirmvalue::TV, firmvalue::TV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, signal::Signal; φmax = one(T)) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes

    indices = logitinteriorindices(grid, qspace)
    Θ = linear_interp(grid.nodes, welfare.P; extrap = constextrap)

    @inbounds Threads.@threads for idx in indices
        i, j, k = idx.I
        a = abatementspace[i]
        z = logitspace[j]    
        q = qspace[k]
        τ = welfare.P[i, j]
        z′ = z + logitdrift(q, τ, τᶜ, signal)

        res = optimize(φ -> firmobjective(φ, a, z′, Θ, τᶜ, firmvalue, qspace, grid, firm, signal), zero(T), φmax, brent)

        nextfirmvalue.V[i, j, k] = e(a, firm) * q + Optim.minimum(res)
        nextfirmvalue.P[i, j, k] = Optim.minimizer(res)
    end

    return nextfirmvalue
end

function solvefirm!(firmvalue::TV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, signal::Signal; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    
    nextfirmvalue = copy(firmvalue)
    for iter in 1:maxiter
        firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, grid, qspace, firm, signal; iterkwargs...)
        
        εᵥ = maximum(abs, nextfirmvalue.V .- firmvalue.V)
        εₚ = maximum(abs, nextfirmvalue.P .- firmvalue.P)
        
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

function governmentstep!(nextwelfare::TW, welfare::TW, firmvalue::TV, τᶜ, grid::G, qspace, firm::Firm, government::Government, signal::Signal; τmax = T(100.0), τgridpoints = 101) where {T, TW <: ValueFunction{2, T}, TV <: ValueFunction{3, T}, G <: Grid{2}}
    abatementspace, logitspace = grid.nodes

    W = linear_interp(grid.nodes, welfare.V; extrap = ConstExtrap())
    Φ = linear_interp((abatementspace, logitspace, qspace), firmvalue.P; extrap = ConstExtrap())

    τgrid = range(zero(T), τmax, τgridpoints)
    indices = logitinteriorindices(grid)

    @inbounds Threads.@threads for idx in indices
        i, j = idx.I
        a = abatementspace[i]
        z = logitspace[j]
        
        τopt = welfare.P[i, j]
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

function solvegovernment!(welfare::TW, firmvalue::TV, τᶜ, grid::G, qspace, firm::Firm, signal::Signal, government::Government; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TW <: ValueFunction{2, T}, TV <: ValueFunction{3, T}, G <: Grid{2}}

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

function nestedpfi!(firmvalue::TV, welfare::TW, τᶜ, grid::G, qspace, firm::Firm, government::Government, signal::Signal; maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0, firmparams = Dict{Symbol, T}(), welfareparams = Dict{Symbol, T}()) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    
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
