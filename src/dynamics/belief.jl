function χ(τ, τᶜ, signal::Signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ)
end

function beliefdrift(χ, φ)
    -φ^2 * (1 - φ) * χ^2
end

function beliefdiffusion(χ, φ)
    φ * (1 - φ) * χ
end
