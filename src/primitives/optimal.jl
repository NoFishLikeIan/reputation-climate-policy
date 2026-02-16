function committedtax(government::Government, firm::Firm)
	@unpack ξ, y₀, δ = government
	@unpack ν, e₀ = firm

	β = e₀^2 / ν

	return (y₀ * ξ * e₀ * β) / (y₀ * ξ * β^2 + β + δ)
end

"Optimal tax ratio `η`, such that `τ = η(φ, z) τᶜ`"
function η(φ, z, signal::Signal, government::Government, firm::Firm)
	@unpack α, ϵ, σ = signal
	@unpack δ = government
	@unpack ν, e₀ = firm

	β = α * e₀ / ν
	δfc = ifelse(δ > 0, δ * (σ / ϵ)^2 / z, zero(φ))

	return (1 + β * φ) / (2 + δfc - β * (1 - φ))
end;

function abatementbestresponse(τ, φ, government::Government, firm::Firm)
	τᶜ = committedtax(government, firm)
	return firm.e₀ * (φ * τᶜ + (1 - φ) * τ) / firm.ν
end

function wᵒ(φ, z, signal::Signal, government::Government, firm::Firm)
	τᶜ = committedtax(government, firm)
	τ = η(φ, z, signal, government, firm) * τᶜ
	a = abatementbestresponse(τ, φ, government, firm)

	return w(τ, a, government, firm)
end