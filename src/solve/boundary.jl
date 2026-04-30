## z → -∞
### Firm
v̲(e, q, ::AbstractFirm) = e * q
φ̲(e, _, ::AbstractFirm, ::Signal) = zero(e)

### Government
τ̲(e, ::AbstractFirm, ::Government) = zero(e)

function w̲(e, firm::Firm, government::Government)
    @unpack e₀, δ = firm
    @unpack β, ξ = government
    ρ = 1 - δ
    a = e₀ - e

    return (ξ / 2) * (e₀^2 / (1 - β) - 2e₀ * a / (1 - β * ρ) + a^2 / (1 - β * ρ^2))
end

function w̲(e, ::FirmPermanentInvestment, government::Government)
    return d(e, government) / (1 - government.β)
end

## z → ∞
### Firm
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

function m̄₁(τᶜ, firm::AbstractFirm, signal::Signal)
    v̄₁(τᶜ, firm, signal) + signal.μ * τᶜ
end

function φ̄(τᶜ, firm::AbstractFirm, signal::Signal)
    @unpack β, κ, ν = firm

    return max((β * m̄₁(τᶜ, firm, signal) - κ) / ν, 0)
end

function φ̄(e, τᶜ, firm::Firm, signal::Signal)
    min(φ̄(τᶜ, firm, signal), investmentupper(e, firm))
end

function S₀(r, N) # ∑_{t=0}^{N-1} r^t
    if N == 0
        return zero(r)
    elseif abs(r - one(r)) <= sqrt(eps(typeof(r))) * max(one(r), abs(r))
        return N * one(r)
    end

    return (one(r) - r^N) / (one(r) - r)
end

function S₁(r, N) # ∑_{t=0}^{N-1} t r^t
    if N <= 1
        return zero(r)
    elseif abs(r - one(r)) <= sqrt(eps(typeof(r))) * max(one(r), abs(r))
        return (N * (N - 1) / 2) * one(r)
    end

    return (r - N * r^N + (N - 1) * r^(N + 1)) / (one(r) - r)^2
end

function S₂(r, N) # ∑_{t=0}^{N-1} t^2 r^t
    if N <= 1
        return zero(r)
    elseif abs(r - one(r)) <= sqrt(eps(typeof(r))) * max(one(r), abs(r))
        return (N * (N - 1) * (2 * N - 1) / 6) * one(r)
    end

    return (
        r * (one(r) + r) - N^2 * r^N +
        (2 * N^2 - 2 * N - 1) * r^(N + 1) -
        (N - 1)^2 * r^(N + 2)
    ) / (one(r) - r)^3
end

function permanentparameters(τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    @unpack β, κ, ν = firm
    m = signal.μ * τᶜ
    B = β * m / (1 - β)
    φstar = max((B - κ) / ν, zero(m))

    return m, B, φstar
end

function permanenthorizon(e, φstar, β)
    scaledemissions = e / φstar
    N = 1
    upperbound = one(scaledemissions) - β

    while scaledemissions > upperbound
        N += 1
        upperbound = N - β * S₀(β, N)
    end

    return N
end

function permanentkkt(e, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    @unpack β, ν = firm
    m, B, φstar = permanentparameters(τᶜ, firm, signal)
    tol = sqrt(eps(typeof(e))) * max(one(e), firm.e₀)

    if e <= tol || φstar <= tol
        return 0, m, B, φstar, zero(e + φstar), one(e + φstar)
    end

    N = permanenthorizon(e, φstar, β)
    inverseβsum = S₀(inv(β), N)
    λ = ν * (N * φstar - e) / inverseβsum

    return N, m, B, φstar, λ, inverseβsum
end

function φ̄(e, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    N, _, _, φstar, λ, _ = permanentkkt(e, τᶜ, firm, signal)

    if N == 0
        return zero(e)
    end

    return min(max(φstar - λ / firm.ν, zero(e + φstar)), e)
end

function m̄₀(τᶜ, firm::Firm, signal::Signal)
    @unpack β, δ, e₀ = firm
    φ = φ̄(τᶜ, firm, signal)

    return (c(φ, firm) + (δ * e₀ - φ) * β * m̄₁(τᶜ, firm, signal)) / (1 - β)
end

function v̄(e, q, τᶜ, firm::Firm, signal::Signal)
    m̄₀(τᶜ, firm, signal) + (v̄₁(τᶜ, firm, signal) + q) * e
end

function m̄(e, τᶜ, firm::Firm, signal::Signal)
    m̄₀(τᶜ, firm, signal) + m̄₁(τᶜ, firm, signal) * e
end

function m̄(e, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    @unpack β, ν = firm
    N, m, _, φstar, λ, inverseβsum = permanentkkt(e, τᶜ, firm, signal)

    if N == 0
        return m * e / (1 - β)
    end

    discountsum = S₀(β, N)

    return m * e / (1 - β) + λ^2 * inverseβsum / (2 * ν) - (ν / 2) * φstar^2 * discountsum
end

function v̄(e, q, τᶜ, firm::FirmPermanentInvestment, signal::Signal)
    φ = φ̄(e, τᶜ, firm, signal)

    return e * q + c(φ, firm) + firm.β * m̄(f(φ, e, firm), τᶜ, firm, signal)
end

### Government
τ̄(τᶜ, ::AbstractFirm, ::Government) = τᶜ

function w̄₂(firm::Firm, government::Government)
    @unpack δ = firm
    @unpack β, ξ = government

    return ξ / (1 - β * (1 - δ)^2)
end
function w̄₂(::FirmPermanentInvestment, government::Government)
    @unpack β, ξ = government

    return ξ / (1 - β)
end

function w̄₁(τᶜ, firm::Firm, government::Government, signal::Signal)
    @unpack δ, e₀ = firm
    @unpack β = government

    φ = φ̄(τᶜ, firm, signal)
    ρ = 1 - δ
    b = δ * e₀ - φ

    return (β * ρ * b * w̄₂(firm, government)) / (1 - β * ρ)
end

function w̄₀(τᶜ, firm::Firm, government::Government, signal::Signal)
    @unpack δ, e₀ = firm
    @unpack β = government

    φ⁺ = φ̄(τᶜ, firm, signal)
    b = δ * e₀ - φ⁺

    return (c(φ⁺, firm) + β * w̄₁(τᶜ, firm, government, signal) * b + (β / 2) * w̄₂(firm, government) * b^2) / (1 - β)
end

function w̄(e, τᶜ, firm::Firm, government::Government, signal::Signal)
    w̄₀(τᶜ, firm, government, signal) + w̄₁(τᶜ, firm, government, signal) * e + w̄₂(firm, government) * e^2 / 2
end

function w̄(e, τᶜ, firm::FirmPermanentInvestment, government::Government, signal::Signal)
    N, _, _, φstar, λ, _ = permanentkkt(e, τᶜ, firm, signal)

    if N == 0
        return d(e, government) / (1 - government.β)
    end

    β = firm.β
    κ = firm.κ
    ν = firm.ν
    βg = government.β
    ξ = government.ξ

    discountsum = S₀(βg, N)
    timeweightsum = S₁(βg, N)
    timeweightedsquaresum = S₂(βg, N)
    inverseweightsum = S₀(βg / β, N)
    timeinverseweightsum = S₁(βg / β, N)
    squaredinverseweightsum = S₀(βg / β^2, N)

    λoverν = λ / ν
    inversecoefficient = λoverν * β / (1 - β)
    constant = e - inversecoefficient
    damages = (ξ / 2) * (
        constant^2 * discountsum -
        2 * constant * φstar * timeweightsum +
        φstar^2 * timeweightedsquaresum +
        2 * constant * inversecoefficient * inverseweightsum -
        2 * φstar * inversecoefficient * timeinverseweightsum +
        inversecoefficient^2 * squaredinverseweightsum
    )
    costs = c(φstar, firm) * discountsum -
        λoverν * (κ + ν * φstar) * inverseweightsum +
        (ν / 2) * λoverν^2 * squaredinverseweightsum

    return damages + costs
end

## Instantiate Value Function
@inline function evaluatefirmcontinuationvalue(V::LI, e, z, q, continuationgrid::G, τᶜ, firm::AbstractFirm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    emissionsclamped = clampnode(e, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return v̲(e, qclamped, firm)
    elseif z >= zmax
        return v̄(e, qclamped, τᶜ, firm, signal)
    end

    return V((emissionsclamped, z, qclamped))
end

@inline function evaluatefirmpolicy(Φ::LI, e, z, q, continuationgrid::G, τᶜ, firm::AbstractFirm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    emissionsclamped = clampnode(e, continuationgrid, 1)
    qclamped = clampnode(q, continuationgrid, 3)
    zmin = zlower(continuationgrid)
    zmax = zupper(continuationgrid)

    if z <= zmin
        return φ̲(e, qclamped, firm, signal)
    elseif z >= zmax
        return φ̄(e, τᶜ, firm, signal)
    end

    return Φ((emissionsclamped, z, qclamped))
end

@inline function evaluatefirmexantevalue(M::LI, e, z, exantegrid::G, τᶜ, firm::AbstractFirm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    emissionsclamped = clampnode(e, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return zero(e)
    elseif z >= zmax
        return m̄(e, τᶜ, firm, signal)
    end

    return M((emissionsclamped, z))
end

@inline function evaluatewelfarevalue(W::LI, e, z, exantegrid::G, τᶜ, firm::AbstractFirm, government::Government, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
    emissionsclamped = clampnode(e, exantegrid, 1)
    zmin = zlower(exantegrid)
    zmax = zupper(exantegrid)

    if z <= zmin
        return w̲(e, firm, government)
    elseif z >= zmax
        return w̄(e, τᶜ, firm, government, signal)
    end

    return W((emissionsclamped, z))
end
