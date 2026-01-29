function committedtax(government::Government, firm::Firm)
	@unpack ξ, δ = government
	@unpack ν = firm

	num = ξ * y₀ * (e₀^3 / ν)
	den = (e₀^4 / ν^2) * y₀ * ξ + (e₀^2 / ν) + δ

	return max((num / den) / taxfactor, ν / e₀) * taxfactor
end

function bestresponsetax(a, z, signal::Signal, government::Government, firm::Firm)
	@unpack α, σ = signal
	@unpack δ = government

	τᶜ = committedtax(government, firm)
	
	return max((α * τᶜ +  a) / (2α - (σ^2 * δ) / (z * α)), 0)
end

function optimaltax(φ, z, signal::Signal, government::Government, firm::Firm)
	@unpack α, σ = signal
	@unpack ν, e₀ = firm
	@unpack δ = government

	τᶜ = committedtax(government, firm)

	num = α + e₀ * φ / ν
	den = 2α - (σ^2 * δ) / (z * α) - e₀ * (1 - φ) / ν

	return (num / den) * τᶜ
end;

function optimalabatement(φ, z, signal::Signal, government::Government, firm::Firm)
	τ = optimaltax(φ, z, signal, government, firm)
	τᶜ = committedtax(government, firm)

	τᵉ = φ * τᶜ + (1 - φ) * τ

	return min((firm.e₀ / firm.ν) * τᵉ, 1)
end