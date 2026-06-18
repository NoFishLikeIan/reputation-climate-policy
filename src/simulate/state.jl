function χ(τ, τᶜ, signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ) 
end

function dφ(φ, τ, τᶜ, signal)
    return -χ(τ, τᶜ, signal)^2 * φ^2 * (1 - φ)
end

function dm(φ, τ, τᶜ, firm)
    e(aᵇ(τ, φ, τᶜ, firm), firm)
end


function F!(dx, x, parameters, _)
    τ, τᶜ, firm, signal = parameters
    
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(φ, m)
    
    dx[1] = dφ(φ, τₜ, τᶜₜ, signal)
    dx[2] = dm(φ, τₜ, τᶜₜ, firm)

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
