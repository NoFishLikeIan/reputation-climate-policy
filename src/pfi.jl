const sqrt2 = √2
const sqrtπ = √π

"Expected firm value, mixture integral over next-period signal s' and believed policy p′."
function expectedfirmvalue(valuefunction::TV, a′, z′, τ′, τᶜ, signal::Signal, grid::G) where {T, TV <: ValueFunction{3, T}, G <: Grid{2}}
    signalspace, signalweights = signal.space
    nodes = (grid.nodes[1], grid.nodes[2], signalspace)
    
    p′ = logistic(z′)
    EV = zero(T)
    
    @inbounds for (k, s) in enumerate(signalspace)
        sⁿᶜ = signal.μ * τ′ + sqrt2 * signal.σ * s
        sᶜ  = signal.μ * τᶜ + sqrt2 * signal.σ * s
        vⁿᶜ = linear_interp(nodes, valuefunction.V, (a′, z′, sⁿᶜ); extrap = ConstExtrap())
        vᶜ  = linear_interp(nodes, valuefunction.V, (a′, z′, sᶜ); extrap = ConstExtrap())
        EV  += signalweights[k] * ((1 - p′) * vⁿᶜ + p′ * vᶜ)
    end

    return EV / sqrtπ
end

function firmstep!(nextfirmvalue::TV, firmvalue::TV, welfare::TW, τᶜ::Real, signal::Signal, grid::G, firm::Firm; φmax = one(T)) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    signalspace, _ = signal.space
    abatementspace, logitspace = grid.nodes
    Θ = linear_interp(grid.nodes, welfare.P; extrap = ConstExtrap())

    @inbounds for (k, s) in enumerate(signalspace), (j, z) in enumerate(logitspace), (i, a) in enumerate(abatementspace)
        τ = welfare.P[i, j]
        z′ = z + logitdrift(s, τ, τᶜ, signal)

        firmobjective = @closure φ -> begin
            a′ = f(φ, a, firm)
            τ′ = Θ((a′, z′))
            return c(φ, firm) + firm.β * expectedfirmvalue(firmvalue, a′, z′, τ′, τᶜ, signal, grid)
        end

        res = optimize(firmobjective, zero(T), φmax, Brent())

        nextfirmvalue.V[i, j, k] = -a * s + Optim.minimum(res)
        nextfirmvalue.P[i, j, k] = Optim.minimizer(res)
    end

    return nextfirmvalue
end

function solvefirm!(firmvalue::TV, welfare::TW, τᶜ::Real, signal::Signal, grid::G, firm::Firm; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    nextfirmvalue = copy(firmvalue)

    for iter in 1:maxiter
        firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, signal, grid, firm; iterkwargs...)
        
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
        @warn @sprintf "Failed firm iteration after %d iterations\n" maxiter
    end

    return maxiter, firmvalue
end

# Government Bellman sweep: w(a,z) = min_{tau>=0} INT [c(phi*(s))+d(e(a'))+beta*w(a',z')] N(s;mu*tau,sig^2) ds
function governmentstep!(nextwelfare::TW, welfare::TW, firmvalue::TV, τᶜ, signal::Signal, grid::G, firm::Firm, government::Government; τmax = 100.0) where {T, TW <: ValueFunction{2, T}, TV <: ValueFunction{3, T}, G <: Grid{2}}
    signalspace, signalweights = signal.space
    abatementspace, logitspace = grid.nodes

    W = linear_interp(grid.nodes, welfare.V; extrap = ConstExtrap())
    Φ = linear_interp((abatementspace, logitspace, signalspace), firmvalue.P; extrap = ConstExtrap())

    @inbounds for (j, z) in enumerate(logitspace), (i, a) in enumerate(abatementspace)
        governmentobjective = @closure τ -> begin
            EV = zero(T)

            for (k, ξ) in enumerate(signalspace)
                s = signal.μ * τ + sqrt2 * signal.σ * ξ
                φ = Φ((a, z, s))
                a′ = f(φ, a, firm)
                z′ = z + logitdrift(s, τ, τᶜ, signal)
                EV += signalweights[k] * (c(φ, firm) + d(e(a, firm), government) + government.β * W((a′, z′)))
            end

            EV / sqrtπ
        end

        res = optimize(governmentobjective, zero(T), τmax, Brent())

        nextwelfare.V[i, j] = Optim.minimum(res)
        nextwelfare.P[i, j] = Optim.minimizer(res)
    end

    return nextwelfare
end

function solvegovernment!(welfare::TW, firmvalue::TV, τᶜ, signal::Signal, grid::G, firm::Firm, government::Government; maxiter = 500, valtol = 1e-8, poltol = 1e-4, verbose = 0, iterkwargs...) where {T, TW <: ValueFunction{2, T}, TV <: ValueFunction{3, T}, G <: Grid{2}}
    nextwelfare = copy(welfare)

    for iter in 1:maxiter
        governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, signal, grid, firm, government; iterkwargs...)

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
        @warn @sprintf "Failed government iteration after %d iterations\n" maxiter
    end

    return maxiter, welfare
end

function nestedpfi!(firmvalue::TV, welfare::TW, τᶜ::Real, τmax::Real, signal::Signal, grid::G, firm::Firm, gov::Government; maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0) where {T, TV <: ValueFunction{3, T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    oldV = similar(welfare.V)
    oldP = similar(welfare.P)

    for iter in 1:maxiter
        copyto!(oldV, welfare.V)
        copyto!(oldP, welfare.P)

        firmiter, _ = solvefirm!(firmvalue, welfare, τᶜ, signal, grid, firm; verbose)
        goviter, _ = solvegovernment!(welfare, firmvalue, τᶜ, signal, grid, firm, gov; verbose)

        εᵥ = maximum(abs, welfare.V .- oldV)
        εₚ = maximum(abs, welfare.P .- oldP)

        if verbose > 0
            @printf "Nested iteration %d, firm iters = %d, gov iters = %d, value error = %.2e, policy error %.2e\n" iter firmiter goviter εᵥ εₚ
        end

        if (εᵥ < valtol && εₚ < poltol)
            return iter, firmvalue, welfare
        end
    end

    if verbose > 0
        @warn @sprintf "Failed nested PFI after %d iterations\n" maxiter
    end

    return maxiter, firmvalue, welfare
end