function constructpolicies(solution, parameters::NonCommittedParameters, grid::NonCommittedGrid)
    n = length(solution.t)
    investment = Array{Float64}(undef, size(grid)..., n)
    tax = similar(investment)
    policies = (; tax, investment)

    return constructpolicies!(policies, solution, parameters)
end
function constructpolicies!(policies, solution, parameters::NonCommittedParameters)
    for (i, s) in enumerate(solution.t)
        policystate = solution(s)
        timepolicy = noncommittedpolicies(policystate, parameters, s)

        policies.tax[:, :, :, i] .= timepolicy.tax
        policies.investment[:, :, :, i] .= timepolicy.investment
    end

    return policies
end

function policy(t, x, policies, parameters::NonCommittedParameters, grid::NonCommittedGrid; extrap = Itp.ClampExtrap())
    s = noncommittedreversetime(t, parameters)
    φ, m, a = x
    
    ninterp = size(policies.tax, 4)
    sspace = range(0, 1, ninterp)

    τᶜₜ = parameters.τᶜ(t)
    τₜ = Itp.linear_interp((grid.φgrid, grid.mgrid, grid.agrid, sspace), policies.tax, (φ, m, a, s); extrap)
    uₜ = Itp.linear_interp((grid.φgrid, grid.mgrid, grid.agrid, sspace), policies.investment, (φ, m, a, s); extrap)  

    return (τₜ, τᶜₜ, uₜ)
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