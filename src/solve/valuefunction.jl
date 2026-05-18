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

function belief(ℓ)
    inv(1 + exp(-ℓ))
end

function logit(φ)
    log(φ / (1 - φ))
end

function Flogit!(dx, x, parameters, ℓ)
    τᶜ, signal, government, firm = parameters
	u, z = x

	@unpack σ, ϵ = signal
	@unpack r = government

    φ = belief(ℓ)
	ηᵩ = ηᵉ(φ, z, τᶜ, signal, government, firm)
	τ = ηᵩ * τᶜ
	a = aᵇ(τ, φ, τᶜ, firm)

	ε = σ / (ϵ * (τᶜ - τ))

	dx[1] = -r * z
	dx[2] = z + 2 * ε^2 * (w(τ, a, government, firm) - u)
end

function leftboundaryexponent(parameters)
    τᶜ, signal, government, firm = parameters

    @unpack σ, ϵ = signal
    @unpack r, δ, y₀, ξ = government
    @unpack e₀, ν = firm

    if δ <= 0
        return one(r)
    end

    A = (σ / (ϵ * τᶜ))^2
    ηz = y₀ * (ϵ / σ)^2 / (δ * e₀^2)
    wa = -y₀ * ξ * e₀
    m = wa * τᶜ * ηz / ν

    b = 1 + 2 * A * m
    α = (b + sqrt(b^2 + 8 * A * r)) / 2

    return min(one(α), α)
end

function leftboundary!(resid, u, parameters)
    _, _, government, firm = parameters
    α = leftboundaryexponent(parameters)
    
    resid[1] = u[1] + government.r * u[2] / α - w(0., 0., government, firm)
end

function rightboundary!(resid, u, parameters)
    τᶜ, _, government, firm = parameters
    
    resid[1] = u[1] - government.r * u[2] - w(0., aᶜ(τᶜ, firm), government, firm)
end
