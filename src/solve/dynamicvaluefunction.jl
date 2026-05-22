"Government's welfare costs as φ → 0"
function leftcost(t, τᶜ, government::Government, firm::DynamicFirm)
    upperτ = ν(t, firm) * firm.e₀
    result = Optim.optimize(τ -> w(t, τ, aᵇ(t, τ, zero(τ), τᶜ, firm), government, firm), zero(upperτ), upperτ)

    return Optim.minimum(result)
end

"Government's welfare costs as φ → 1"
function rightcost(t, τᶜ, government::Government, firm::DynamicFirm)
    return w(t, zero(τᶜ), aᶜ(t, τᶜ, firm), government, firm)
end

"Finite difference stencil"
@inline function gridderivativestencil(i, n)
    if (i == 2) && (n > 5)
        i:(i + 3)
    elseif (i == n - 1) && (n > 5)
        (i - 3):i
    else
        (i - 1):(i + 1)
    end
end

function gridderivativeweights(ℓgrid, stencil, node, ℓ)
    denominator = one(ℓ)

    for k in stencil
        k == node && continue
        denominator *= ℓgrid[node] - ℓgrid[k]
    end

    firstweight = zero(ℓ)
    secondweight = zero(ℓ)

    for j in stencil
        j == node && continue

        firstterm = one(ℓ)
        for k in stencil
            (k == node || k == j) && continue
            firstterm *= ℓ - ℓgrid[k]
        end
        firstweight += firstterm

        for m in stencil
            (m == node || m == j) && continue

            secondterm = one(ℓ)
            for k in stencil
                (k == node || k == j || k == m) && continue
                secondterm *= ℓ - ℓgrid[k]
            end
            secondweight += secondterm
        end
    end

    return firstweight / denominator, secondweight / denominator
end

function gridderivatives(u, ℓgrid, i)
    n = length(ℓgrid)
    stencil = gridderivativestencil(i, n)
    ℓ = ℓgrid[i]
    uₗ = zero(u[i])
    uₗₗ = zero(u[i])

    for j in stencil
        firstweight, secondweight = gridderivativeweights(ℓgrid, stencil, j, ℓ)
        uₗ += firstweight * u[j]
        uₗₗ += secondweight * u[j]
    end

    return uₗ, uₗₗ
end

function dynamicfoc(t, φ, η, uₗ, uₗₗ, τᶜ, signal::Signal, government::Government, firm::DynamicFirm)
    νₜ = ν(t, firm)
    taxshare = φ + (1 - φ) * η

    abatementeffect = (1 - φ) / νₜ * (
        -government.y₀ * government.ξ * firm.e₀
        + (1 + government.y₀ * government.ξ / νₜ) * τᶜ * taxshare
    )

    asseteffect = government.δ * firm.e₀ * τᶜ * η / government.y₀ * (
        firm.e₀ - τᶜ / νₜ * (φ + 3 * (1 - φ) * η / 2)
    )

    reputationeffect = -(signal.ϵ / signal.σ)^2 * τᶜ * (1 - η) * (uₗₗ - uₗ) / government.r

    return abatementeffect + asseteffect + reputationeffect
end

function dynamichamiltonian(t, φ, η, uₗ, uₗₗ, τᶜ, signal::Signal, government::Government, firm::DynamicFirm)
    τ = η * τᶜ
    a = aᵇ(t, τ, φ, τᶜ, firm)
    ξ = signal.ϵ * (τᶜ - τ) / signal.σ

    return government.r * w(t, τ, a, government, firm) + ξ^2 * (uₗₗ - uₗ) / 2
end

function ηᵈ(t, φ, uₗ::T, uₗₗ, τᶜ, signal::Signal, government::Government, firm::DynamicFirm) where T <: Real
    ηlow = zero(T)
    ηhigh = one(T)

    if τᶜ ≤ 0
        return ηlow
    end

    if τᶜ ≥ ν(t, firm) * firm.e₀
        result = Optim.optimize(
            η -> dynamichamiltonian(t, φ, η, uₗ, uₗₗ, τᶜ, signal, government, firm),
            ηlow,
            ηhigh,
        )

        return Optim.minimizer(result)
    end

    foc = @closure η -> dynamicfoc(t, φ, η, uₗ, uₗₗ, τᶜ, signal, government, firm)

    if foc(ηlow) ≥ 0
        return ηlow
    elseif foc(ηhigh) ≤ 0
        return ηhigh
    else
        return Roots.find_zero(foc, (ηlow, ηhigh))
    end
end

function dynamicHJB!(du, u, parameters, t)
    ℓgrid, tgrid, τᶜgrid, signal, government, firm = parameters

    τᶜ = linear_interp(tgrid, τᶜgrid, t)

    du[1] = government.r * (u[1] - leftcost(t, τᶜ, government, firm))
    du[end] = government.r * (u[end] - rightcost(t, τᶜ, government, firm))

    n = length(ℓgrid)

    @inbounds for i in 2:(n - 1)
        φ = belief(ℓgrid[i])
        uₗ, uₗₗ = gridderivatives(u, ℓgrid, i)

        τ = ηᵈ(t, φ, uₗ, uₗₗ, τᶜ, signal, government, firm) * τᶜ
        a = aᵇ(t, τ, φ, τᶜ, firm)
        ξ = signal.ϵ * (τᶜ - τ) / signal.σ

        du[i] = government.r * u[i] - government.r * w(t, τ, a, government, firm) - ξ^2 * (uₗₗ - uₗ) / 2
    end

    return du
end
