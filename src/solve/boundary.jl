function v̲(a, q, firm::AbstractFirm)
    return e(a, firm) * q
end

function φ̲(a, _, ::AbstractFirm, ::Signal)
    zero(a)
end

function τ̲(a, ::AbstractFirm, ::Government)
    zero(a)
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
function v̄₁(τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    @unpack β = firm
    @unpack μ = signal

    return (β * μ * τᶜ) / (1 - β)
end

function v̄ᵉ(τᶜ, firm::AbstractFirm, signal::Signal)
    v̄₁(τᶜ, firm, signal) + signal.μ * τᶜ
end

function φ̄(τᶜ, firm::AbstractFirm, signal::Signal)
    @unpack β, κ, ν = firm

    return max((β * v̄ᵉ(τᶜ, firm, signal) - κ) / ν, 0)
end

function v̄₀(τᶜ, firm::AbstractFirm, signal::Signal)
    @unpack β, e₀ = firm
    @unpack μ = signal
    φ = φ̄(τᶜ, firm, signal)

    return (β * μ * τᶜ * e₀ + c(φ, firm) - β * v̄ᵉ(τᶜ, firm, signal) * φ) / (1 - β)
end

function ψ̄(a, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    @unpack β, e₀ = firm
    m = signal.μ * τᶜ
    flow = φ̄(τᶜ, firm, signal)
    a = clamp(a, zero(a), e₀)
    x = e₀ - a
    tol = sqrt(eps(typeof(x))) * max(one(x), e₀)

    if flow <= tol
        return m * x / (1 - β)
    end

    value = zero(x + m + flow)
    discount = one(value)

    while x > tol
        φ = min(flow, x)
        value += discount * (m * x + c(φ, firm))
        x -= φ
        discount *= β
    end

    return value
end

function ψ̄(a, τᶜ, firm::Firm, signal::Signal)
    v̄₀(τᶜ, firm, signal) + signal.μ * τᶜ * e(a, firm) - v̄₁(τᶜ, firm, signal) * a
end

function v̄(a, q, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    φ = φ̄(a, τᶜ, firm, signal)

    return e(a, firm) * q + c(φ, firm) + firm.β * ψ̄(f(φ, a, firm), τᶜ, firm, signal)
end

function v̄(a, q, τᶜ, firm::Firm, signal::Signal)
    v̄₀(τᶜ, firm, signal) + e(a, firm) * q - v̄₁(τᶜ, firm, signal) * a
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

function w̄(a, τᶜ, firm::FirmPermanentInvestment, government::Government, signal::Signal)
    @unpack e₀ = firm
    @unpack β = government
    flow = φ̄(τᶜ, firm, signal)
    a = clamp(a, zero(a), e₀)
    x = e₀ - a
    tol = sqrt(eps(typeof(x))) * max(one(x), e₀)

    if flow <= tol
        return d(x, government) / (1 - β)
    end

    value = zero(x + flow)
    discount = one(value)

    while x > tol
        φ = min(flow, x)
        value += discount * (d(x, government) + c(φ, firm))
        x -= φ
        discount *= β
    end

    return value
end

@inline function evaluatefirmcontinuationvalue(V::LI, a, z, q, continuationgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return v̲(a, qclamped, firm)
    elseif z >= zmax
        return v̄(a, qclamped, τᶜ, firm, signal)
    end

    return V((aclamped, z, qclamped))
end

@inline function evaluatefirmpolicy(Φ::LI, a, z, q, continuationgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return φ̲(a, qclamped, firm, signal)
    elseif z >= zmax
        return φ̄(a, τᶜ, firm, signal)
    end

    return Φ((aclamped, z, qclamped))
end

@inline function evaluatefirmexantevalue(Ψ::LI, a, z, exantegrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    aclamped = clampnode(a, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return zero(a)
    elseif z >= zmax
        return ψ̄(a, τᶜ, firm, signal)
    end

    return Ψ((aclamped, z))
end

@inline function evaluatewelfarevalue(W::LI, a, z, exantegrid::G, τᶜ, firm::Firm, government::Government, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    aclamped = clampnode(a, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return w̲(a, firm, government)
    elseif z >= zmax
        return w̄(a, τᶜ, firm, government, signal)
    end

    return W((aclamped, z))
end
