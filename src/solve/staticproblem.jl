function staticinitialguess(parameters, φ)
    τᶜ, _, government, firm = parameters

    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)

    ∂u = u₁ - u₀
    α = leftboundaryexponent(parameters)

    û = u₀ + ∂u * φ^α
    ẑ = -α * ∂u * φ^α * (1 - φ) / government.r

    return [û, ẑ]
end

function fillstaticguess!(guessarray, i, φ, u, z, parameters)
    guessarray[i, 1] = u
    guessarray[i, 2] = z

    if size(guessarray, 2) == 4
        s, rhsu, rhsz = staticrightside(parameters, φ, u, z)
        guessarray[i, 3] = rhsu / s
        guessarray[i, 4] = rhsz / s
    end
end

function initialiseguess(parameters, ε::T, guessnodes, statevariables) where T

    φnodes = range(ε, 1 - ε, guessnodes)

    guessarray = Matrix{T}(undef, guessnodes, statevariables)
    @inbounds for i in eachindex(φnodes)
        u, z = staticinitialguess(parameters, φnodes[i])
        fillstaticguess!(guessarray, i, φnodes[i], u, z, parameters)
    end

    return φnodes, guessarray
end

function updateguess!(guessarray, φnodes, solution, ε, ε′, parameters)
    n = length(φnodes)
    τᶜ, _, government, firm = parameters
    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)
    α = leftboundaryexponent(parameters)

    leftz = solution(ε)[2]
    rightz = solution(1 - ε)[2]

    nextφnodes  = range(ε′, 1 - ε′, n)

    for (i, φ) in enumerate(nextφnodes)
        if φ < ε
            z = leftz * (φ / ε)^α
            u = u₀ - government.r * z / α
        elseif φ > 1 - ε
            z = rightz * (1 - φ) / ε
            u = u₁ + government.r * z
        else
            u, z = solution(φ)[1:2]
        end

        fillstaticguess!(guessarray, i, φ, u, z, parameters)
    end

    return nextφnodes, guessarray
end

const defaultεs = [1e-2];
function solvestatic(τᶜ, signal::Signal, government::Government, firm::StaticFirm; εs = defaultεs, verbose = false, guessnodes = 100, φstepfactor = 1e-3, endpointstepfactor = 0.2, algorithm = nothing, solvekwargs...)
    parameters = (τᶜ, signal, government, firm)
    bcresiduals = (zeros(1), zeros(1))
    solutions = Tuple{Float64, Vector{Float64}, Vector{Vector{Float64}}}[]
    εpath = sort(collect(εs); rev = true)
    n = length(εpath)

    φnodes, guessarray = initialiseguess(parameters, εpath[1], guessnodes, 2)

    for (i, ε) in enumerate(εpath)

        if verbose
            @printf "Solving ε continuation %d/%d with ε = %.2e\n" i n ε
        end

        dφ = min((1 - 2ε) * φstepfactor, endpointstepfactor * ε)

        guess = @closure (_, φ) -> FastInterpolations.linear_interp(φnodes, FastInterpolations.Series(guessarray), φ; extrap = FastInterpolations.ClampExtrap())

        prob = BVP.TwoPointBVProblem(F!, (leftboundary!, rightboundary!), guess, (ε, 1 - ε), parameters; bcresid_prototype = bcresiduals)

        solvealgorithm = isnothing(algorithm) ? BVP.MIRK6() : algorithm
        solution = BVP.solve(prob, solvealgorithm; dt = dφ, progress = verbose, solvekwargs...)

        if !SciMLBase.successful_retcode(solution)
            if verbose
                @error @sprintf "BVP with φ = %.2e failed with error: %s" ε solution.retcode
            end
            break
        end

        push!(solutions, (ε, solution.t, [x[1:2] for x in solution.u]))

        if i < n
            φnodes, guessarray = updateguess!(guessarray, φnodes, solution, ε, εpath[i + 1], parameters)
        end
    end

    return solutions
end

function solvestaticmassmatrix(τᶜ, signal::Signal, government::Government, firm::StaticFirm; εs = defaultεs, verbose = false, guessnodes = 100, φstepfactor = 1e-3, endpointstepfactor = 0.2, algorithm = nothing, solvekwargs...)
    parameters = (τᶜ, signal, government, firm)
    bcresiduals = (zeros(1), zeros(1))
    solutions = Tuple{Float64, Vector{Float64}, Vector{Vector{Float64}}}[]
    εpath = sort(collect(εs); rev = true)
    n = length(εpath)
    M = [
        1. 0. 0. 0.
        0. 1. 0. 0.
        0. 0. 0. 0.
        0. 0. 0. 0.
    ]

    φnodes, guessarray = initialiseguess(parameters, εpath[1], guessnodes, 4)

    for (i, ε) in enumerate(εpath)

        if verbose
            @printf "Solving mass-matrix ε continuation %d/%d with ε = %.2e\n" i n ε
        end

        dφ = min((1 - 2ε) * φstepfactor, endpointstepfactor * ε)
        guess = @closure (_, φ) -> FastInterpolations.linear_interp(φnodes, FastInterpolations.Series(guessarray), φ; extrap = FastInterpolations.ClampExtrap())

        bvpfunction = BVP.BVPFunction(Fmass!, (leftboundary!, rightboundary!); mass_matrix = M, twopoint = Val(true), bcresid_prototype = bcresiduals)
        prob = BVP.TwoPointBVProblem(bvpfunction, guess, (ε, 1 - ε), parameters)

        solvealgorithm = isnothing(algorithm) ? BVP.Ascher4() : algorithm
        solution = BVP.solve(prob, solvealgorithm; dt = dφ, progress = verbose, solvekwargs...)

        if !SciMLBase.successful_retcode(solution)
            if verbose
                @error @sprintf "Mass-matrix BVP with φ = %.2e failed with error: %s" ε solution.retcode
            end
            break
        end

        push!(solutions, (ε, solution.t, [x[1:2] for x in solution.u]))

        if i < n
            φnodes, guessarray = updateguess!(guessarray, φnodes, solution, ε, εpath[i + 1], parameters)
        end
    end

    return solutions
end
