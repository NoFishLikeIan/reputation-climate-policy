function F!(dx, x, parameters, φ)
    τᶜ, signal, government, firm = parameters
	u, z = x

	@unpack σ, ϵ = signal
	@unpack r = government

	ηᵩ = ηᵉ(φ, z, τᶜ, signal, government, firm)
	τ = ηᵩ * τᶜ
	a = aᵇ(τ, φ, τᶜ, firm)

	s = φ * (1 - φ)
	ε = σ / (ϵ * (τᶜ - τ))
	
	du = -r * z
	dz = z + 2 * ε^2 * (w(τ, a, government, firm) - u)

	dx[1] = du / s
	dx[2] = dz / s
end

function leftboundary!(resid, u, parameters)
    _, _, government, firm = parameters
    
    resid[1] = u[1] - w(0., 0., government, firm)
end

function rightboundary!(resid, u, parameters)
    τᶜ, _, government, firm = parameters
    
    resid[2] = u[2] - w(τᶜ, aᶜ(τᶜ, firm), government, firm)
end