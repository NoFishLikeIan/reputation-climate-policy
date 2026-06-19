function staticrightside(parameters, φ, u, z)
    τᶜ, signal, government, firm = parameters

    @unpack σ, ϵ = signal
    @unpack r = government

    ηratio = ηᵉ(φ, z, τᶜ, signal, government, firm)
    τ = ηratio * τᶜ
    a = aᵇ(τ, φ, τᶜ, firm)

    s = φ * (1 - φ)
    inverseprecision = σ / (ϵ * τᶜ * (1 - ηratio))

    rhsu = -r * z
    rhsz = z + 2 * inverseprecision^2 * (w(τ, a, government, firm) - u)

    return s, rhsu, rhsz
end

function F!(dx, x, parameters, φ)
	u, z = x

	s, rhsu, rhsz = staticrightside(parameters, φ, u, z)

	dx[1] = rhsu / s
	dx[2] = rhsz / s
end

function belief(ℓ)
    inv(1 + exp(-ℓ))
end

function logit(φ)
    log(φ / (1 - φ))
end

function Flogit!(dx, x, parameters, ℓ)
	u, z = x

    φ = belief(ℓ)
	_, rhsu, rhsz = staticrightside(parameters, φ, u, z)

	dx[1] = rhsu
	dx[2] = rhsz
end

function Fmass!(dx, x, parameters, φ)
    u, z, uφ, zφ = x

    s, rhsu, rhsz = staticrightside(parameters, φ, u, z)

    dx[1] = uφ
    dx[2] = zφ
    dx[3] = s * uφ - rhsu
    dx[4] = s * zφ - rhsz
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
    τᶜ, signal, government, firm = parameters

    @unpack σ, ϵ = signal
    @unpack r, δ, y₀, ξ = government
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
    _, _, government, firm = parameters
    α = leftboundaryexponent(parameters)
    
    resid[1] = u[1] + government.r * u[2] / α - w(0., 0., government, firm)
end

function rightboundary!(resid, u, parameters)
    τᶜ, _, government, firm = parameters
    
    resid[1] = u[1] - government.r * u[2] - w(0., aᶜ(τᶜ, firm), government, firm)
end
