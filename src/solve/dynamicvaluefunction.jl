function linearinterpolate(x, xs, ys)
    if x <= first(xs)
        return first(ys)
    elseif x >= last(xs)
        return last(ys)
    end

    i = searchsortedlast(xs, x)
    weight = (x - xs[i]) / (xs[i + 1] - xs[i])

    return (1 - weight) * ys[i] + weight * ys[i + 1]
end

function τᶜpath(t, tgrid, τᶜgrid)
    linearinterpolate(t, tgrid, τᶜgrid)
end

function leftcost(t, τᶜ, government::Government, firm::DynamicFirm, sunkabatement::Bool)
    if sunkabatement
        return w(t, zero(τᶜ), zero(τᶜ), government, firm)
    end

    upperτ = ν(t, firm) * firm.e₀
    result = Optim.optimize(τ -> w(t, τ, aᵇ(t, τ, zero(τ), τᶜ, firm), government, firm), zero(upperτ), upperτ)

    return Optim.minimum(result)
end

function rightcost(t, τᶜ, government::Government, firm::DynamicFirm)
    return w(t, zero(τᶜ), aᶜ(t, τᶜ, firm), government, firm)
end

function ηᵈ(t, φ, uℓ, uℓℓ, τᶜ, signal::Signal, government::Government, firm::DynamicFirm)
    if τᶜ <= 0
        return zero(τᶜ)
    end

    objective = η -> begin
        τ = η * τᶜ
        a = aᵇ(t, τ, φ, τᶜ, firm)
        ξ = signal.ϵ * (τᶜ - τ) / signal.σ

        return government.r * w(t, τ, a, government, firm) + ξ^2 * (uℓℓ - uℓ) / 2
    end

    result = Optim.optimize(objective, zero(τᶜ), one(τᶜ))

    return Optim.minimizer(result)
end

function dynamicHJB!(du, u, parameters, t)
    ℓgrid, tgrid, τᶜgrid, signal, government, firm, sunkabatement = parameters

    τᶜ = τᶜpath(t, tgrid, τᶜgrid)
    dℓ = ℓgrid[2] - ℓgrid[1]

    du[1] = government.r * (u[1] - leftcost(t, τᶜ, government, firm, sunkabatement))
    du[end] = government.r * (u[end] - rightcost(t, τᶜ, government, firm))

    for i in 2:(length(ℓgrid) - 1)
        φ = belief(ℓgrid[i])
        uℓ = (u[i + 1] - u[i - 1]) / (2dℓ)
        uℓℓ = (u[i + 1] - 2u[i] + u[i - 1]) / dℓ^2

        η = ηᵈ(t, φ, uℓ, uℓℓ, τᶜ, signal, government, firm)
        τ = η * τᶜ
        a = aᵇ(t, τ, φ, τᶜ, firm)
        ξ = signal.ϵ * (τᶜ - τ) / signal.σ

        du[i] = government.r * u[i] - government.r * w(t, τ, a, government, firm) - ξ^2 * (uℓℓ - uℓ) / 2
    end

    return nothing
end

function terminalcondition(terminaltime, τᶜ, signal::Signal, government::Government, firm::DynamicFirm; kwargs...)
    terminalfirm = StaticFirm(terminaltime, firm)

    solution = solvestaticproblem(
        τᶜ,
        signal,
        government,
        terminalfirm;
        kwargs...
    )

end
