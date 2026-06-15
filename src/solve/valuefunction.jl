function F!(dx, x, parameters, φ)
    τᶜ, signal, climate, government, firm = parameters
	u, z = x

	@unpack σ, ϵ = signal
	@unpack r = government

	ηᵩ = ηᵉ(φ, z, τᶜ, signal, government, firm)
	τ = ηᵩ * τᶜ
	a = aᵇ(τ, φ, τᶜ, firm)
    m = e(a, firm) / climate.δₘ

	s = φ * (1 - φ)
	ε = σ / (ϵ * (τᶜ - τ))
	
	du = -r * z
	dz = z + 2 * ε^2 * (w(m, τ, a, climate, government, firm) - u)

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
    τᶜ, signal, climate, government, firm = parameters
	u, z = x

	@unpack σ, ϵ = signal
	@unpack r = government

    φ = belief(ℓ)
	τ = ηᵉ(φ, z, τᶜ, signal, government, firm) * τᶜ
	a = aᵇ(τ, φ, τᶜ, firm)
    m = e(a, firm) / climate.δₘ

	ε = σ / (ϵ * (τᶜ - τ))

	dx[1] = -r * z
	dx[2] = z + 2 * ε^2 * (w(m, τ, a, climate, government, firm) - u)
end

function positivequadraticroot(b, c)
    discriminantroot = hypot(b, sqrt(c))

    if b ≥ 0
        return (b + discriminantroot) / 2
    else
        return c / (2 * (discriminantroot - b))
    end
end

function leftboundaryexponent(parameters)
    τᶜ, signal, climate, government, firm = parameters

    @unpack σ, ϵ = signal
    @unpack r, δ, y₀ = government
    @unpack ξ = climate
    @unpack e₀, ν = firm

    if δ ≤ 0
        return one(r)
    end

    A = (σ / (ϵ * τᶜ))^2
    ηz = y₀ * (ϵ / σ)^2 / (δ * e₀^2)
    wa = -y₀ * ξ * e₀
    m = wa * τᶜ * ηz / ν

    b = 1 + 2 * A * m
    α = positivequadraticroot(b, 8 * A * r)

    return min(1, α)
end

function leftboundary!(resid, u, parameters)
    _, _, climate, government, firm = parameters
    α = leftboundaryexponent(parameters)
    a = 0.
    m = e(a, firm) / climate.δₘ
    
    resid[1] = u[1] + government.r * u[2] / α - w(m, 0., a, climate, government, firm)
end

function rightboundary!(resid, u, parameters)
    τᶜ, _, climate, government, firm = parameters
    τ = 0.
    a = aᶜ(τᶜ, firm)
    m = e(a, firm) / climate.δₘ
    
    resid[1] = u[1] - government.r * u[2] - w(m, τ, a, climate, government, firm)
end
