function κ(φ, z, signal::Signal, government::Government, firm::Firm)	
	@unpack σ, ϵ = signal
	τᶜ = committedtax(government, firm)

	σ / (ϵ * (1 - η(φ, z, signal, government, firm)) * τᶜ)
end;

κ²(φ, z, signal, government, firm) = κ(φ, z, signal, government, firm)^2

function κ²(φ, z, model)
	signal, government, firm = model
	return κ²(φ, z, signal, government, firm)
end

function wᵒ(φ, z, model)
	signal, government, firm = model
	return wᵒ(φ, z, signal, government, firm)
end

function F(x, model, φ)
	government = model[2]
	u, z = x

	du = government.r * z
	dz = z + 2κ²(φ, z, model) * (u - wᵒ(φ, z, model))

	return SVector{2}(du, dz) / (φ * (1 - φ))
end
