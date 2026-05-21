function staticinitialguess(parameters, ℓ)
    τᶜ, _, government, firm = parameters

    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)

    ∂u = u₁ - u₀
    α = leftboundaryexponent(parameters)
    φ = belief(ℓ)

    û = u₀ + ∂u * φ^α
    ẑ = -α * ∂u * φ^α * (1 - φ) / government.r

    return [û, ẑ]
end

function staticcontinuationguess(previoussol, previousstep, parameters)
    τᶜ, _, government, firm = parameters
    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)
    α = leftboundaryexponent(parameters)

    leftx = previoussol(logit(previousstep))
    rightx = previoussol(logit(1 - previousstep))
    leftz = leftx[2]
    rightz = rightx[2]

    return (_, ℓ) -> begin
        φ = belief(ℓ)

        if φ < previousstep
            ẑ = leftz * (φ / previousstep)^α
            û = u₀ - government.r * ẑ / α
            return [û, ẑ]
        elseif φ > 1 - previousstep
            ẑ = rightz * (1 - φ) / previousstep
            û = u₁ + government.r * ẑ
            return [û, ẑ]
        else
            return previoussol(ℓ)
        end
    end
end

function solvestaticproblem(τᶜ, signal::Signal, government::Government, firm::StaticFirm; φsteps = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4], verbose = true, solvekwargs...)
    parameters = (τᶜ, signal, government, firm)
    bcresiduals = (zeros(1), zeros(1))
    solutions = Tuple{Float64, Vector{Float64}, Vector{Vector{Float64}}}[]
    solution = nothing
    guess = staticinitialguess

    for (i, φstep) in enumerate(φsteps)
        if verbose
            @printf "Solving problem %d/%d with φ = %.2e\n" i length(φsteps) φstep
        end

        ℓspan = logit.((φstep, 1 - φstep))
        ℓstep = min(0.05, (last(ℓspan) - first(ℓspan)) / 200)

        prob = TwoPointBVProblem(
            Flogit!,
            (leftboundary!, rightboundary!),
            guess,
            ℓspan,
            parameters;
            bcresid_prototype = bcresiduals,
        )
        
        sol = solve(prob, MIRK4(); dt = ℓstep, solvekwargs...)

        if verbose && !successful_retcode(sol.retcode)
            @warn "Static BVP failed at φstep = $(φstep) with retcode $(sol.retcode)"
        end

        push!(solutions, (φstep, sol.t, sol.u))
        solution = sol
        guess = staticcontinuationguess(sol, φstep, parameters)
    end

    return (solutions = solutions, solution = solution)
end
