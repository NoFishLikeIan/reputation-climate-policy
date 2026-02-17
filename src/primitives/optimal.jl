function abatement(τ, firm::F) where F <: AbstractFirm
	(firm.e₀ / firm.ν) * τ
end;

function wᶜ(τ, government, firm)
	w(τ, abatement(τ, firm), government, firm)
end;

function optimaltax(a, z, signal::Signal, firm::EnergyFirm)
	@unpack α, ϵ, σ = signal
	@unpack δ = firm

	s = (ϵ / σ)^2
	
	num = τᶜ + α * a
	den = 2 + e(a, firm)^2 / (2δ * s * z)
	
	return num / den
end;

function optimaltax(a, signal::Signal, firm::InelasticEnergyFirm)
	@unpack α, ϵ, σ = signal
	
	return (τᶜ + α * a) / 2
end;

function equilibriumcondition(τ, z, φ, signal::Signal, firm::F) where F <: AbstractFirm
	τᵉ = φ * τᶜ + (1 - φ) * τ
	a = abatement(τᵉ, firm)
	return optimaltax(a, z, signal, firm) - τ
end;

function equilibriumtax(z, φ, signal::Signal, firm::EnergyFirm; atol = 1e-5, τbounds = (zero(z), 1.5one(z)))
	find_zero(τ -> equilibriumcondition(τ, z, φ, signal, firm), τbounds; atol)
end