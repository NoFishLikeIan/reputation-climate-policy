function wᶜ(m, τᶜ, climate::Climate, government::Government, firm::Firm)
    w(m, τᶜ, a(τᶜ, firm), climate, government, firm)
end

function optimalcommittedtax(∂ₘu, government::Government, firm::Firm)
    @unpack δ, y₀, r = government
    @unpack e₀, ν = firm

    upperτ = firm.e₀ * firm.ν

    if ∂ₘu ≤ 0
        return zero(upperτ)
    end

    if δ ≤ 0
        return clamp(∂ₘu / (r * y₀), zero(upperτ), upperτ)
    end

    linear = y₀ + δ * e₀^2 * ν / y₀
    discriminant = linear^2 - 6δ * e₀ * ∂ₘu / (r * y₀)

    if discriminant < 0
        return upperτ
    end

    τ = y₀ * (linear - √discriminant) / (3δ * e₀)

    return clamp(τ, zero(upperτ), upperτ)
end

function L(m, τ, a, z, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    reputationvalue = z * μ(τ, signal) * (μ(τᶜ, signal) - μ(τ, signal)) / signal.σ^2

    return w(m, τ, a, climate, government, firm) - reputationvalue
end

function b(z, signal::Signal, government::Government, firm::Firm)
	@unpack ϵ, σ = signal
	@unpack δ, y₀ = government
	@unpack e₀ = firm

	return (δ * e₀ / y₀) / (z * (ϵ / σ)^2)
end

function η(a, z, signal::Signal, government::Government, firm::Firm)	
    inv(2 + b(z, signal, government, firm) * e(a, firm))
end;

function ηᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    if z ≤ 0
		return zero(z)
	end
	
	ā = (firm.e₀ - φ * τᶜ / firm.ν)
	bᶻ = b(z, signal, government, firm)
	x = ā * bᶻ
	d = (2 + x)^2 - 4bᶻ * (1 - φ) * τᶜ / firm.ν
	
	return 1 / (1 + (x + √d) / 2)
end

function τᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    ηᵉ(φ, z, τᶜ, signal, government, firm) * τᶜ
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
