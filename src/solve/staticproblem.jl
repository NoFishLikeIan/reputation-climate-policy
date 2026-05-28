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

function staticνcontinuationguess(previoussol, previousparameters, parameters)
    previousτᶜ, _, previousgovernment, previousfirm = previousparameters
    τᶜ, _, government, firm = parameters

    previousu₀ = w(0., 0., previousgovernment, previousfirm)
    previousu₁ = w(0., aᶜ(previousτᶜ, previousfirm), previousgovernment, previousfirm)
    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)

    previousrange = previousu₁ - previousu₀
    valuerange = u₁ - u₀
    valuescale = iszero(previousrange) ? one(valuerange) : valuerange / previousrange

    return (_, ℓ) -> begin
        previousu, previousz = previoussol(ℓ)

        û = u₀ + valuescale * (previousu - previousu₀)
        ẑ = valuescale * previousz

        return [û, ẑ]
    end
end

const defaultφsteps = [1e-2, 1e-3, 1e-4];

function defaultνsteps(firm::StaticFirm; νstart = ν₀, νnodes = 10)
    νtarget = firm.ν

    if νtarget <= 0 || νstart <= 0
        error("ν continuation requires positive ν values.")
    end

    if νnodes <= 1 || νstart ≈ νtarget
        return [νtarget]
    end

    return collect(exp.(range(log(νstart), log(νtarget), length = νnodes)))
end

function νcontinuationpath(νsteps, firm::StaticFirm)
    νpath = collect(νsteps)

    if isempty(νpath)
        push!(νpath, firm.ν)
    elseif !(last(νpath) ≈ firm.ν)
        push!(νpath, firm.ν)
    end

    if any(ν -> ν <= 0, νpath)
        error("ν continuation requires positive ν values.")
    end

    return νpath
end

function solvestaticproblemdata(τᶜ, signal::Signal, government::Government, firm::StaticFirm; φsteps = defaultφsteps, verbose = false, ℓstepfactor = 5e-3, initialguess = staticinitialguess, solvekwargs...)
    parameters = (τᶜ, signal, government, firm)
    bcresiduals = (zeros(1), zeros(1))
    solutions = Tuple{Float64, Vector{Float64}, Vector{Vector{Float64}}}[]

    guess = initialguess
    n = length(φsteps)
    sol = nothing

    for (i, φstep) in enumerate(φsteps)
        if verbose
            @printf "Solving problem %d/%d with φ = %.2e\n" i n φstep
        end

        ℓspan = logit.((φstep, 1 - φstep))
        ℓstep = 2ℓspan[end] * ℓstepfactor

        prob = BVP.TwoPointBVProblem(Flogit!, (leftboundary!, rightboundary!), guess, ℓspan, parameters; bcresid_prototype = bcresiduals)

        sol = BVP.solve(prob, BVP.MIRK6(); dt = ℓstep, progress = verbose, solvekwargs...)

        if verbose && !SciMLBase.successful_retcode(sol)
            @warn @sprintf "BVP with φ = %.2e failed with error: %s" φstep sol.retcode
        end

        push!(solutions, (φstep, sol.t, sol.u))
        guess = staticcontinuationguess(sol, φstep, parameters)
    end

    return solutions, sol
end

function solvestaticproblem(τᶜ, signal::Signal, government::Government, firm::StaticFirm; φsteps = defaultφsteps, verbose = false, ℓstepfactor = 5e-3, initialguess = staticinitialguess, solvekwargs...)
    solutions, _ = solvestaticproblemdata(
        τᶜ,
        signal,
        government,
        firm;
        φsteps,
        verbose,
        ℓstepfactor,
        initialguess,
        solvekwargs...
    )

    return solutions
end

function solvestaticνcontinuation(τᶜ, signal::Signal, government::Government, firm::StaticFirm; νsteps = defaultνsteps(firm), φsteps = defaultφsteps, verbose = false, ℓstepfactor = 5e-3, solvekwargs...)
    νpath = νcontinuationpath(νsteps, firm)
    νsolutions = NamedTuple[]
    previoussol = nothing
    previousparameters = nothing
    n = length(νpath)

    for (i, νstep) in enumerate(νpath)
        stepfirm = StaticFirm(e₀ = firm.e₀, ν = νstep)
        stepτᶜ = i == n ? τᶜ : computeτᶜ(government, stepfirm)
        parameters = (stepτᶜ, signal, government, stepfirm)
        initialguess = previoussol === nothing ? staticinitialguess : staticνcontinuationguess(previoussol, previousparameters, parameters)

        if verbose
            @printf "Solving ν continuation %d/%d with ν = %.3e\n" i n νstep
        end

        solutions, sol = solvestaticproblemdata(
            stepτᶜ,
            signal,
            government,
            stepfirm;
            φsteps,
            verbose,
            ℓstepfactor,
            initialguess,
            solvekwargs...
        )

        push!(νsolutions, (ν = νstep, τᶜ = stepτᶜ, solutions = solutions))
        previoussol = sol
        previousparameters = parameters
    end

    return νsolutions
end
