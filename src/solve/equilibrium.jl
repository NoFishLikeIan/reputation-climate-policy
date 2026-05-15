function wᶜ(τ, government::Government, firm::Firm)
    w(τ, aᶜ(τ, firm), government, firm)
end

function computeτᶜ(government::Government, firm::Firm)
    upperτ = firm.ν * firm.e₀
    result = Optim.optimize(τ -> wᶜ(τ, government, firm), zero(upperτ), upperτ)

    return Optim.minimizer(result)
end

function L(τ, a, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    reputationvalue = z * μ(τ, signal) * (μ(τᶜ, signal) - μ(τ, signal)) / signal.σ^2

    return w(τ, a, government, firm) - reputationvalue
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
    if z < 1e-3
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

function wᵉ(φ, z, τᶜ, signal::Signal, government::Government, firm::Firm)
    τ = τᵉ(φ, z, τᶜ, signal, government, firm)
    a = aᵉ(τ, φ, τᶜ, firm)

    return w(τ, a, government, firm)
end
