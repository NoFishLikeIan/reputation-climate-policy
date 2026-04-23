const extrapstrategy = ExtendExtrap()

@inline normalizederror(εᵥ, εₚ, valtol, poltol) = max(εᵥ / valtol, εₚ / poltol)

@inline function errorincreased(ε, prevε, worsetol)
    isfinite(prevε) || return false
    ε > prevε * (one(prevε) + oftype(prevε, worsetol))
end

@inline function insufficientimprovement(ε, prevε, improvetol)
    isfinite(prevε) || return false
    prevε <= zero(prevε) && return false

    prevε - ε <= prevε * oftype(prevε, improvetol)
end

function firmstep!(newboundary::TV, firmboundary::TV, τᶜ, abatementspace, firm::Firm, signal::Signal; φmax = 1000one(T)) where {T, TV <: ValueFunction{1, T}}

    @inbounds Threads.@threads for i in eachindex(abatementspace)
        a = abatementspace[i]
        res = Optim.optimize(φ -> firmcosts(φ, a, τᶜ, firmboundary.V, abatementspace, firm, signal), zero(T), φmax, Brent())
        newboundary.V[i] = Optim.minimum(res)
        newboundary.P[i] = Optim.minimizer(res)
    end

    return newboundary
end

function boundarypfi!(firmboundary::TV, τᶜ, abatementspace, firm::Firm, signal::Signal; maxiter = 100, valtol = 1e-8, poltol = 1e-4, improvetol = 1e-3, worsetol = 1e-3, maxstall = 5, verbose = 0) where {T, TV <: ValueFunction{1, T}}
    oldboundary = copy(firmboundary)
    prevε = T(Inf)
    stalliter = 0

    for iter in 1:maxiter
        copyto!(oldboundary, firmboundary)

        firmstep!(firmboundary, oldboundary, τᶜ, abatementspace, firm, signal)
        εᵥ = maximum(abs, oldboundary.V .- firmboundary.V)
        εₚ = maximum(abs, oldboundary.P .- firmboundary.P)
        ε = normalizederror(εᵥ, εₚ, valtol, poltol)

        if verbose > 0
            @printf "Boundary iteration %d, value error = %.2e, policy error %.2e\r" iter εᵥ εₚ
        end

        if ε < one(ε)
            return iter, firmboundary
        elseif errorincreased(ε, prevε, worsetol)
            copyto!(firmboundary, oldboundary)

            if verbose > 0
                @warn @sprintf "Boundary policy iteration stopped after %d iterations because the normalized error increased\n" iter
            end

            return iter, firmboundary
        elseif insufficientimprovement(ε, prevε, improvetol)
            stalliter += 1

            if stalliter >= maxstall
                if verbose > 0
                    @warn @sprintf "Boundary policy iteration stopped after %d iterations because the normalized error improved too slowly\n" iter
                end

                return iter, firmboundary
            end
        else
            stalliter = 0
        end

        prevε = ε
    end

    if verbose > 0
        @warn @sprintf "Boundary policy iteration not converged after %d iterations\n" maxiter
    end

    return maxiter, firmboundary
end

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

@inline function evaluatefirmexpostvalue(V::LI, a, z, q, expostgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, expostgrid, 1)
    qclamped = clampnode(q, expostgrid, 3)
    zmin = zlower(expostgrid)
    zmax = zupper(expostgrid)

    if z <= zmin
        return v̲(aclamped, qclamped, firm)
    elseif z >= zmax
        return v̄(aclamped, qclamped, τᶜ, firm, signal)
    end

    return V((aclamped, z, qclamped))
end

@inline function evaluatefirmpolicy(Φ::LI, a, z, q, expostgrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{3}}
    aclamped = clampnode(a, expostgrid, 1)
    qclamped = clampnode(q, expostgrid, 3)
    zmin = zlower(expostgrid)
    zmax = zupper(expostgrid)

    if z <= zmin
        return φ̲(aclamped, qclamped, firm, signal)
    elseif z >= zmax
        return φ̄(τᶜ, firm, signal)
    end

    return Φ((aclamped, z, qclamped))
end

@inline function evaluatefirmcontinuation(Ψ::LI, a, z, exantegrid::G, τᶜ, firm::Firm, signal::Signal) where {LI <: FastInterpolations.AbstractInterpolant, G <: Grid{2}}
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
