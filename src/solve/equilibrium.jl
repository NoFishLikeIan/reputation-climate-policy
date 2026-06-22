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

    result = Optim.optimize(obj, zero(T), τmax, brent)
    
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

function τᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τmax = clamp(τᶜ, zero(τᶜ), firm.ν * firm.e₀)

    if z ≤ 0 || τmax ≤ 0
        return zero(τmax)
    end

    obj = τ -> begin
        aᵢ = aᵇ(τ, φ, τᶜ, firm)
        reputationvalue = z * μ(τ, signal) * (μ(τᶜ, signal) - μ(τ, signal)) / signal.σ^2
        government.y₀ * c(aᵢ, firm) + l(aᵢ, τ, government, firm) - reputationvalue
    end

    result = Optim.optimize(obj, zero(τmax), τmax, Optim.Brent())
    return Optim.minimizer(result)
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
