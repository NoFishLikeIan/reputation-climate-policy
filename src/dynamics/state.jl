function χ(τ, τᶜ, signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ) 
end

function beliefdrift(χ, φ)
    -φ^2 * (1 - φ) * χ^2
end

function beliefdiffusion(χ, φ)
    φ * (1 - φ) * χ
end

function F!(dx, x, parameters, _)
    τ, τᶜ, firm, signal = parameters
    
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(x)
    
    dx[1] = beliefdrift(χ(τₜ, τᶜₜ, signal), φ)
    dx[2] = e(aᵇ(τₜ, φ, τᶜₜ, firm), firm)

    return dx
end

function G!(Σ, x, parameters, _)

    τ, τᶜ, _, signal = parameters
    
    φ, m = x


    Σ[1, 1] = beliefdiffusion(χ(τ(x), τᶜ(m), signal), φ)

end
