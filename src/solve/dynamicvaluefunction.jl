"Government's welfare costs as φ → 0"
function leftcost(t, τᶜ, government::Government, firm::DynamicFirm)
    return w(t, zero(τᶜ), zero(τᶜ), government, firm)
end

"Government's welfare costs as φ → 1"
function rightcost(t, τᶜ, government::Government, firm::DynamicFirm)
    return w(t, zero(τᶜ), aᶜ(t, τᶜ, firm), government, firm)
end

const dynamicpdeclampextrap = FastInterpolations.ClampExtrap()
const dynamicpdecontext = Ref{Any}()

function discountedcontinuation(tgrid, costs, r)
    values = similar(costs)
    values[end] = costs[end]

    @inbounds for i in (length(tgrid) - 1):-1:1
        discount = exp(-r * (tgrid[i + 1] - tgrid[i]))
        values[i] = (1 - discount) * costs[i] + discount * values[i + 1]
    end

    return values
end

function setdynamicpdecontext!(terminaltime, φgrid, terminalu, tgrid, τᶜgrid, signal::Signal, government::Government, firm::DynamicFirm)
    leftcostgrid = [leftcost(t, τᶜ, government, firm) for (t, τᶜ) in zip(tgrid, τᶜgrid)]
    rightcostgrid = [rightcost(t, τᶜ, government, firm) for (t, τᶜ) in zip(tgrid, τᶜgrid)]

    dynamicpdecontext[] = (
        terminaltime = terminaltime,
        φgrid = collect(φgrid),
        terminalu = collect(terminalu),
        tgrid = collect(tgrid),
        τᶜgrid = collect(τᶜgrid),
        leftvalue = discountedcontinuation(tgrid, leftcostgrid, government.r),
        rightvalue = discountedcontinuation(tgrid, rightcostgrid, government.r),
        signal = signal,
        government = government,
        firm = firm,
    )

    return dynamicpdecontext[]
end

function dynamicpdecalendar(s)
    context = dynamicpdecontext[]
    return context.terminaltime - s
end

function dynamicpdetax(t)
    context = dynamicpdecontext[]
    return FastInterpolations.linear_interp(context.tgrid, context.τᶜgrid, t; extrap = dynamicpdeclampextrap)
end

function dynamicpdeterminal(φ)
    context = dynamicpdecontext[]
    return FastInterpolations.linear_interp(context.φgrid, context.terminalu, φ; extrap = dynamicpdeclampextrap)
end

function dynamicpdeleftvalue(s)
    context = dynamicpdecontext[]
    t = dynamicpdecalendar(s)
    return FastInterpolations.linear_interp(context.tgrid, context.leftvalue, t; extrap = dynamicpdeclampextrap)
end

function dynamicpderightvalue(s)
    context = dynamicpdecontext[]
    t = dynamicpdecalendar(s)
    return FastInterpolations.linear_interp(context.tgrid, context.rightvalue, t; extrap = dynamicpdeclampextrap)
end

function dynamicpdeflow(s, φ, u, uφ, uφφ)
    context = dynamicpdecontext[]
    t = dynamicpdecalendar(s)
    τᶜ = dynamicpdetax(t)
    φdrift = φ * (1 - φ)
    generatorcurvature = -φ * uφ + φdrift * uφφ / 2
    D = 2 * φdrift * generatorcurvature
    τ = ηᵈ(t, φ, D, τᶜ, context.signal, context.government, context.firm) * τᶜ
    a = aᵇ(t, τ, φ, τᶜ, context.firm)
    ξ = context.signal.ϵ * (τᶜ - τ) / context.signal.σ

    return context.government.r * w(t, τ, a, context.government, context.firm) + φdrift * ξ^2 * generatorcurvature - context.government.r * u
end

"Finite difference stencil"
@inline function derivativestencil(i, n)
    if (i <= 2) && (n > 5)
        1:4
    elseif (i >= n - 1) && (n > 5)
        (n - 3):n
    elseif i == 1
        1:3
    elseif i == n
        (n - 2):n
    else
        (i - 1):(i + 1)
    end
end

function derivativeweight(ℓgrid, stencil, node, ℓ)
    denominator = one(ℓ)
    weight = zero(ℓ)

    for k in stencil
        k == node && continue
        denominator *= ℓgrid[node] - ℓgrid[k]
    end

    for j in stencil
        j == node && continue

        firstterm = one(ℓ)
        for k in stencil
            (k == node || k == j) && continue
            firstterm *= ℓ - ℓgrid[k]
        end
        weight += firstterm
    end

    return weight / denominator
end

function derivative(u, ℓgrid, i)
    n = length(ℓgrid)
    stencil = derivativestencil(i, n)
    ℓ = ℓgrid[i]
    uₗ = zero(u[i])

    for j in stencil
        weight = derivativeweight(ℓgrid, stencil, j, ℓ)
        uₗ += weight * u[j]
    end

    return uₗ
end

function dynamicfoc(t, φ, η, D, τᶜ, signal::Signal, government::Government, firm::DynamicFirm)
    τ = η * τᶜ

    a = aᵇ(t, τ, φ, τᶜ, firm)

    asseteffect = government.δ * firm.e₀ * τᶜ * η * e(a, firm) / government.y₀

    reputationeffect = -(signal.ϵ / signal.σ)^2 * τᶜ * (1 - η) * D / government.r

    return asseteffect + reputationeffect
end

function ηᵈ(t, φ, D::T, τᶜ, signal::Signal, government::Government, firm::DynamicFirm) where T <: Real
    ηlow = zero(T)
    ηhigh = one(T)

    if τᶜ ≤ 0
        return ηlow
    end

    foc = @closure η -> dynamicfoc(t, φ, η, D, τᶜ, signal, government, firm)

    if foc(ηlow) ≥ 0
        return ηlow
    elseif foc(ηhigh) ≤ 0
        return ηhigh
    else
        return Roots.find_zero(foc, (ηlow, ηhigh))
    end
end

function dynamicHJB!(dx, x, parameters, t)
    ℓgrid, tgrid, τᶜgrid, signal, government, firm = parameters

    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜgrid, t)
    n = length(ℓgrid)
    (size(x, 1) == n && size(x, 2) == 2) || error("Dynamic HJB state must be an $(n) by 2 matrix.")

    u = @view x[:, 1]
    z = @view x[:, 2]
    du = @view dx[:, 1]
    dz = @view dx[:, 2]

    du[1] = government.r * (u[1] - leftcost(t, τᶜ, government, firm))
    du[n] = government.r * (u[n] - rightcost(t, τᶜ, government, firm))
    dz[1] = zero(z[1])
    dz[n] = zero(z[n])

    @inbounds for i in 2:(n - 1)
        φ = belief(ℓgrid[i])
        zₗ = derivative(z, ℓgrid, i)
        D = government.r * (z[i] - zₗ)

        τ = ηᵈ(t, φ, D, τᶜ, signal, government, firm) * τᶜ
        a = aᵇ(t, τ, φ, τᶜ, firm)
        ξ = signal.ϵ * (τᶜ - τ) / signal.σ

        du[i] = government.r * u[i] - government.r * w(t, τ, a, government, firm) - ξ^2 * D / 2
    end

    @inbounds for i in 2:(n - 1)
        duₗ = derivative(du, ℓgrid, i)
        dz[i] = -duₗ / government.r
    end

    return dx
end
