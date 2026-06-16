function χ(τ, τᶜ, signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ) 
end

function F!(dx, x, parameters, _)
    τ, τᶜ, firm, signal = parameters
    
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(φ, m)
    
    dx[1] = χ(τₜ, τᶜₜ, signal)^2 * φ^2 * (1 - φ)
    dx[2] = firm.e₀ - a(φ * τᶜₜ + (1 - φ) * τₜ, firm)

    return dx
end

function G!(Σ, x, parameters, _)

    τ, τᶜ, _, signal = parameters
    
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(φ, m)

    Σ[1, 1] = χ(τₜ, τᶜₜ, signal) * φ * (1 - φ)

    return Σ
end