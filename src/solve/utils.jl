Model = Tuple{Firm, Government, Signal}

function prolongchebyshevcoefficients(coefficients::TC, order) where { T, TC <: AbstractMatrix{T}}
    targetsize = order .+ 1
    prolonged = zeros(T, targetsize)

    oldindices = ntuple(i -> axes(coefficients, i), 2)
    prolonged[oldindices...] .= coefficients

    return prolonged
end

const bobyqa = OptimizationNLopt.NLopt.LN_BOBYQA

function solvechebyshevcontinuation(objectivefunction::SciMLBase.OptimizationFunction, initialpolicy:: FastChebInterp.ChebPoly, orders::TO, lowerbound, upperbound, optparameters;
        coefficientscale = one(eltype(lowerbound)),
        coefficientlower::Real = -Inf,
        coefficientupper::Real = Inf,
        optimizer = bobyqa,
        stoponfailure::Bool = true,
        solvekwargs...
    ) where { TO <: AbstractVector{Tuple{Int, Int}} }



    η = vec(initialpolicy.coefs ./ coefficientscale)
    stages = Any[]

    for (iteration, order) in enumerate(orders)
        if iteration > 1
            previousorder = orders[iteration - 1]
            previouscoefficients = reshape(η, previousorder .+ 1)
            η = vec(prolongchebyshevcoefficients(previouscoefficients, order))
        end

        lower = fill(coefficientlower, size(η))
        upper = fill(coefficientupper, size(η))
        η = clamp.(η, lower, upper)

        problem = SciMLBase.OptimizationProblem(objectivefunction, η, optparameters; lb = lower, ub = upper)
        solution = Optimization.solve(problem, optimizer; solvekwargs...)
        coefficients = reshape(solution.u, order .+ 1)
        policy = FastChebInterp.ChebPoly(coefficientscale .* coefficients, lowerbound, upperbound)
        successful = SciMLBase.successful_retcode(solution)

        stage = (order = order, solution = solution, policy = policy, successful = successful)
        push!(stages, stage)

        η = solution.u

        if stoponfailure && !successful
            @warn "Failed optimisation on order $order"
            break
        end
    end

    finalstage = last(stages)
    converged = length(stages) == length(orders) && all(stage -> stage.successful, stages)

    return (
        policy = finalstage.policy,
        solution = finalstage.solution,
        stages = stages,
        orders = orders,
        converged = converged,
    )
end
