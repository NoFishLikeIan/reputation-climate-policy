struct Policies{Tτ, Tu, Tv}
    tax::Tτ
    investment::Tu
    beliefvalue::Tv
end

const clampextrap = Itp.ClampExtrap()
function constructpolicies(solution, parameters::NonCommittedParameters, grid::NonCommittedGrid; extrap = clampextrap)
    n = length(solution.t)
    investment = Array{Float64}(undef, size(grid)..., n)
    tax = similar(investment)
    beliefvalue = similar(investment)
    sgrid = solution.t

    for (i, s) in enumerate(solution.t)
        policystate = solution(s)
        timepolicy = noncommittedpolicies(policystate, parameters, s)

        tax[:, :, :, i] .= timepolicy.tax
        investment[:, :, :, i] .= timepolicy.investment
        beliefvalue[:, :, :, i] .= timepolicy.beliefvalue
    end

    interpolationspace = (grid.φgrid, grid.mgrid, grid.agrid, sgrid)

    τ = Itp.linear_interp(interpolationspace, tax; extrap)
    u = Itp.linear_interp(interpolationspace, investment; extrap)
    ∂u = Itp.linear_interp(interpolationspace, beliefvalue; extrap)

    return Policies(τ, u, ∂u)
end

function dynamicdrift(x, dynamicparameters, t)
    policies, τᶜ, horizon, models = dynamicparameters
    firm, _, signal, _ = models
    φ, m, a = x
    s = noncommittedreversetime(t, horizon)

    τₜ = policies.tax(φ, m, a, s)
    τᶜₜ = τᶜ(t)
    uₜ = policies.investment(φ, m, a, s)

    dφ = beliefdrift(χ(τₜ, τᶜₜ, signal), φ)
    dm = cumulativeemissionsdrift(a, firm)
    da = uₜ

    return SA.SVector(dφ, dm, da)
end
function dynamicnoise(x, dynamicparameters, t)
    policies, τᶜ, horizon, models = dynamicparameters
    signal = models[3]
    φ, m, a = x
    s = noncommittedreversetime(t, horizon)

    τₜ = policies.tax(φ, m, a, s)
    τᶜₜ = τᶜ(t)

    σᵩ = beliefdiffusion(χ(τₜ, τᶜₜ, signal), φ)

    return SA.SVector(σᵩ, 0, 0)
end

# Log-odds system
logistic(ℓ) = inv(exp(-ℓ) + 1)
function logdynamicdrift(x, dynamicparameters, t)
    policies, τᶜ, horizon, models = dynamicparameters
    firm, _, signal, _ = models
    ℓ, m, a = x
    φ = logistic(ℓ)

    s = noncommittedreversetime(t, horizon)

    τₜ = policies.tax(φ, m, a, s)
    τᶜₜ = τᶜ(t)
    uₜ = policies.investment(φ, m, a, s)
    χₜ = χ(τₜ, τᶜₜ, signal)
    
    dℓ = -χₜ^2 / 2
    dm = cumulativeemissionsdrift(a, firm)
    da = uₜ

    return SA.SVector(dℓ, dm, da)
end
function logdynamicnoise(x, dynamicparameters, t)
    policies, τᶜ, horizon, models = dynamicparameters
    signal = models[3]
    ℓ, m, a = x
    φ = logistic(ℓ)

    s = noncommittedreversetime(t, horizon)

    τₜ = policies.tax(φ, m, a, s)
    τᶜₜ = τᶜ(t)
    χₜ = χ(τₜ, τᶜₜ, signal)
    
    return SA.SVector(χₜ, 0, 0)
end