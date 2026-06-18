function wᶜ(m, τᶜ, climate::Climate, government::Government, firm::Firm)
    w(m, τᶜ, a(τᶜ, firm), climate, government, firm)
end

function optimalcommittedtax(∂ₘu, government::Government{T}, firm::Firm{T}) where T

    if ∂ₘu ≤ 0
        return zero(T)
    end
    
    τmax = firm.e₀ * firm.ν

    obj = @closure τ -> begin
        aᶜ = a(τ, firm)
        government.r * (government.y₀ * c(aᶜ, firm) + l(aᶜ, τ, government, firm)) - aᶜ * ∂ₘu
    end

    result = Optim.optimize(obj, zero(T), τmax, Optim.Brent())
    
    if !Optim.converged(result)
        return T(NaN)
    else
        return Optim.minimizer(result)
    end
end

function L(m, τ, a, z, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    reputationvalue = z * μ(τ, signal) * (μ(τᶜ, signal) - μ(τ, signal)) / signal.σ^2

    return w(m, τ, a, climate, government, firm) - reputationvalue
end

function τbest(a, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τmax = clamp(τᶜ, zero(τᶜ), firm.ν * firm.e₀)

    if z ≤ 0 || τmax ≤ 0
        return zero(τmax)
    elseif !isfinite(z)
        return τmax / 2
    end

    obj = τ -> begin
        reputationvalue = z * μ(τ, signal) * (μ(τᶜ, signal) - μ(τ, signal)) / signal.σ^2
        l(a, τ, government, firm) - reputationvalue
    end

    result = Optim.optimize(obj, zero(τmax), τmax, Optim.Brent())
    return Optim.minimizer(result)
end

function τᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τmax = clamp(τᶜ, zero(τᶜ), firm.ν * firm.e₀)

    if z ≤ 0 || τmax ≤ 0
        return zero(τmax)
    end

    residual = τ -> τbest(aᵇ(τ, φ, τᶜ, firm), z, τᶜ, signal, government, firm) - τ
    low = zero(τmax)
    high = τmax
    lowresidual = residual(low)
    highresidual = residual(high)

    if lowresidual ≤ 0
        return low
    elseif highresidual ≥ 0
        return high
    else
        result = Optim.optimize(τ -> residual(τ)^2, low, high, Optim.Brent())
        return Optim.minimizer(result)
    end
end

function ηᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τᶜ ≤ 0 && return zero(τᶜ)
    τᵉ(φ, z, τᶜ, signal, government, firm) / τᶜ
end

function aᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τ = τᵉ(φ, z, τᶜ, signal, government, firm)
    return aᵇ(τ, φ, τᶜ, firm)
end

function wᵉ(m, φ, z, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    τ = τᵉ(φ, z, τᶜ, signal, government, firm)
    a = aᵉ(φ, z, τᶜ, signal, government, firm)

    return w(m, τ, a, climate, government, firm)
end
