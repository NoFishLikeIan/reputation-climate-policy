@inline normalizederror(εᵥ, εₚ, valtol, poltol) = max(εᵥ / valtol, εₚ / poltol)

function v̲(a, q, firm::Firm)
    return e(a, firm) * q
end

function φ̲(a, q, ::Firm, ::Signal)
    zero(a)
end

function τ̲(a, ::Firm, ::Government)
    return zero(a)
end

function w̲(a, firm::Firm, government::Government)
    @unpack e₀, δ = firm
    @unpack β, ξ = government

    return (ξ / 2) * (e₀^2 / (1 - β) - 2 * e₀ * a / (1 - β * (1 - δ)) + a^2 / (1 - β * (1 - δ)^2))
end

function v̄₁(τᶜ, firm::Firm, signal::Signal)
    @unpack β, δ = firm
    @unpack μ = signal

    return (β * (1 - δ) * μ * τᶜ) / (1 - β * (1 - δ))
end

function v̄ᵉ(τᶜ, firm::Firm, signal::Signal)
    v̄₁(τᶜ, firm, signal) + signal.μ * τᶜ
end

function φ̄(τᶜ, firm::Firm, signal::Signal)
    @unpack β, κ, ν = firm

    return max((β * v̄ᵉ(τᶜ, firm, signal) - κ) / ν, 0)
end

function v̄₀(τᶜ, firm::Firm, signal::Signal)
    @unpack β, e₀ = firm
    @unpack μ = signal
    φ = φ̄(τᶜ, firm, signal)

    return (β * μ * τᶜ * e₀ + c(φ, firm) - β * v̄ᵉ(τᶜ, firm, signal) * φ) / (1 - β)
end

function v̄(a, q, τᶜ, firm::Firm, signal::Signal)
    v̄₀(τᶜ, firm, signal) + e(a, firm) * q - v̄₁(τᶜ, firm, signal) * a
end

function ψ̄(a, τᶜ, firm::Firm, signal::Signal)
    v̄₀(τᶜ, firm, signal) + signal.μ * τᶜ * e(a, firm) - v̄₁(τᶜ, firm, signal) * a
end

function τ̄(τᶜ, ::Firm, ::Government)
    return τᶜ
end

function w̄₂(firm::Firm, government::Government)
    @unpack δ = firm
    @unpack β, ξ = government

    return ξ / (1 - β * (1 - δ)^2)
end

function w̄₁(τᶜ, firm::Firm, government::Government, signal::Signal)
    @unpack δ, e₀ = firm
    @unpack β, ξ = government

    φ⁺ = φ̄(τᶜ, firm, signal)

    return (β * (1 - δ) * φ⁺ * w̄₂(firm, government) - ξ * e₀) / (1 - β * (1 - δ))
end

function w̄₀(τᶜ, firm::Firm, government::Government, signal::Signal)
    @unpack δ, e₀ = firm
    @unpack β, ξ = government

    φ⁺ = φ̄(τᶜ, firm, signal)

    return (c(φ⁺, firm) + (ξ / 2) * e₀^2 + β * w̄₁(τᶜ, firm, government, signal) * φ⁺ + (β / 2) * w̄₂(firm, government) * φ⁺^2) / (1 - β)
end

function w̄(a, τᶜ, firm::Firm, government::Government, signal::Signal)
    w̄₀(τᶜ, firm, government, signal) + w̄₁(τᶜ, firm, government, signal) * a + w̄₂(firm, government) * a^2 / 2
end

@inline function evaluatefirmcontinuationvalue(V::LI, a, z, q, continuationgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return v̲(aclamped, qclamped, firm)
    elseif z >= zmax
        return v̄(aclamped, qclamped, τᶜ, firm, signal)
    end

    return V((aclamped, z, qclamped))
end

@inline function evaluatefirmpolicy(Φ::LI, a, z, q, continuationgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return φ̲(aclamped, qclamped, firm, signal)
    elseif z >= zmax
        return φ̄(τᶜ, firm, signal)
    end

    return Φ((aclamped, z, qclamped))
end

@inline function evaluatefirmexantevalue(Ψ::LI, a, z, exantegrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    aclamped = clampnode(a, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return zero(aclamped)
    elseif z >= zmax
        return ψ̄(aclamped, τᶜ, firm, signal)
    end

    return Ψ((aclamped, z))
end

@inline function evaluatewelfarevalue(W::LI, a, z, exantegrid::G, τᶜ, firm::Firm, government::Government, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    aclamped = clampnode(a, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return w̲(aclamped, firm, government)
    elseif z >= zmax
        return w̄(aclamped, τᶜ, firm, government, signal)
    end

    return W((aclamped, z))
end
