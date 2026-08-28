function constructpolicies(solution, parameters::NonCommittedParameters, grid::NonCommittedGrid)
    n = length(solution.t)
    investment = Array{Float64}(undef, size(grid)..., n)
    tax = similar(investment)
    beliefvalue = similar(investment)
    sgrid = solution.t
    policies = (; tax, investment, beliefvalue, sgrid)

    return constructpolicies!(policies, solution, parameters)
end
function constructpolicies!(policies, solution, parameters::NonCommittedParameters)
    for (i, s) in enumerate(solution.t)
        policystate = solution(s)
        timepolicy = noncommittedpolicies(policystate, parameters, s)

        policies.tax[:, :, :, i] .= timepolicy.tax
        policies.investment[:, :, :, i] .= timepolicy.investment
        policies.beliefvalue[:, :, :, i] .= timepolicy.beliefvalue
    end

    return policies
end

function policy(t, x, policies, parameters::NonCommittedParameters, grid::NonCommittedGrid; extrap = Itp.ClampExtrap())
    s = noncommittedreversetime(t, parameters)
    φ, m, a = x

    τᶜₜ = parameters.τᶜ(t)
    interpolationspace = (grid.φgrid, grid.mgrid, grid.agrid, policies.sgrid)
    state = (φ, m, a, s)

    τₜ = Itp.linear_interp(
        interpolationspace, policies.tax, state; extrap
    )
    uₜ = Itp.linear_interp(
        interpolationspace, policies.investment, state; extrap
    )

    return (τₜ, τᶜₜ, uₜ)
end

function interpolatebeliefvalue(
    t,
    x,
    policies,
    parameters::NonCommittedParameters,
    grid::NonCommittedGrid;
    extrap = Itp.ClampExtrap(),
)
    s = noncommittedreversetime(t, parameters)
    φ, m, a = x

    interpolationspace = (grid.φgrid, grid.mgrid, grid.agrid, policies.sgrid)
    state = (φ, m, a, s)

    return Itp.linear_interp(
        interpolationspace, policies.beliefvalue, state; extrap
    )
end

function dynamicdrift(x, dynamicparameters, t)
    solution, parameters, grid = dynamicparameters
    φ, _, a = x

    τₜ, τᶜₜ, uₜ = policy(t, x, solution, parameters, grid)

    dφ = beliefdrift(χ(τₜ, τᶜₜ, parameters.signal), φ)
    dm = cumulativeemissionsdrift(a, parameters.firm)
    da = uₜ

    return SA.SVector(dφ, dm, da)
end

function dynamicnoise(x, dynamicparameters, t)
    solution, parameters, grid = dynamicparameters
    φ = x[1]

    τₜ, τᶜₜ, _ = policy(t, x, solution, parameters, grid)
    σᵩ = beliefdiffusion(χ(τₜ, τᶜₜ, parameters.signal), φ)
    
    return SA.SVector(σᵩ, 0, 0)
end
