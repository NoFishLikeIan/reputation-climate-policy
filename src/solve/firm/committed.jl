const committedlaguerrequadrature = FastGaussQuadrature.gausslaguerre(32)

function committedboundaryobj(a, m, τᶜ, firm::Firm)
    x = SA.SVector(a, m)
    return τᶜ(x) - firm.r * c(a, firm)
end

"Computes the initial point of investment `aₛ, mₛ` at time `tₛ` of the policy `τᶜ`."
function committedinitialstate(τᶜ, firm::Firm, climate::Climate)
    a₀, m₀ = firm.a₀, climate.m₀
    m̄ = τᶜ.ub[2]
    boundaryobj₀ = committedboundaryobj(a₀, m₀, τᶜ, firm)

    isleftactive = boundaryobj₀ < 0

    if isleftactive
        terminalboundaryobj = committedboundaryobj(a₀, m̄, τᶜ, firm)
        isrightactive = terminalboundaryobj ≥ 0

        if isrightactive
            obj = @closure m -> committedboundaryobj(a₀, m, τᶜ, firm)
            mₛ = brent(obj, (m₀, m̄))
            tₛ = (mₛ - m₀) / e(a₀, firm)

            return a₀, mₛ, tₛ
        else
            t̄ = (m̄ - m₀) / e(a₀, firm)
            return a₀, m̄, t̄
        end
    else
        terminalboundaryobj = committedboundaryobj(firm.e₀, m₀, τᶜ, firm)

        aₛ = if terminalboundaryobj < 0
            obj = @closure a -> committedboundaryobj(a, m₀, τᶜ, firm)
            brent(obj, (a₀, firm.e₀))
        else
            firm.e₀
        end

        return aₛ, m₀, zero(m₀)
    end
end

function stallingcosts(τᶜ::FastChebInterp.ChebPoly{2, TD, TR}, a, m̄, firm::Firm, government::Government, climate::Climate) where {TD, TR}
    points, weights = committedlaguerrequadrature
    Δm = m̄ - climate.m₀
    horizon = government.r * Δm / e(a, firm)

    payoff = zero(TR)
    @inbounds for (xᵢ, wᵢ) in zip(points, weights)
        fraction = -expm1(-xᵢ)
        m = climate.m₀ + Δm * fraction
        tax = τᶜ(SA.SVector(a, m))
        payoff += wᵢ * exp(-horizon * fraction) * w(m, tax, government, climate)
    end

    return horizon * payoff
end

function tailcosts(a, m, t, tax, firm::Firm, government::Government, climate::Climate)
    emissions = e(a, firm)
    discount = exp(-government.r * t)

    if iszero(emissions)
        return discount * w(m, tax, government, climate)
    end

    points, weights = committedlaguerrequadrature
    payoff = zero(promote_type(typeof(tax), typeof(government.r)))

    @inbounds for i in eachindex(points, weights)
        mtail = m + emissions * points[i] / government.r
        payoff += weights[i] * w(mtail, tax, government, climate)
    end

    return discount * payoff
end

function welfaredrift(u, parameters, a)
    τᶜ, firm, government, climate = parameters
    t, m, _ = u
    x = clamp.(SA.SVector(a, m), τᶜ.lb, τᶜ.ub)

    if isnan(x[2])
        println(τᶜ.coefs, "\n\n", ForwardDiff.value.(τᶜ.coefs), "\n\n", ForwardDiff.partials.(τᶜ.coefs), "\n\n", ForwardDiff.value(a), " ", ForwardDiff.value(m))
    end

    τᶜₜ, ∇τᶜₜ = FastChebInterp.chebgradient(τᶜ, x)
    ∂ₐτᶜₜ, ∂ₘτᶜₜ = ∇τᶜₜ

    dm = (firm.r * c′(a, firm) - ∂ₐτᶜₜ) / ∂ₘτᶜₜ
    dt = dm / e(a, firm)

    discount = government.r * exp(-government.r * t)
    dJ₂ = discount * (w(m, τᶜₜ, government, climate) * dt + c(a, firm))

    return SA.SVector(dt, dm, dJ₂)
end

function transitioncosts(τᶜ, aₛ, mₛ, tₛ, ā, firm::Firm, government::Government, climate::Climate)
    if aₛ ≈ ā
        return tₛ, mₛ, zero(tₛ)
    end

    parameters = (τᶜ, firm, government, climate)
    uₛ = SA.SVector(tₛ, mₛ, zero(tₛ))
    welfareprob = ODE.ODEProblem{false}(welfaredrift, uₛ, (aₛ, ā), parameters)
    welfaresol = ODE.solve(welfareprob; save_everystep = false, save_start = false)

    return welfaresol.u[end]
end

"Compute the committed government's annualised cost. The tax is held constant after
the upper cumulative-emissions bound, and the investment boundary is assumed to be
regular and single crossing."
function welfarecosts(τᶜ::FastChebInterp.ChebPoly{2, TD, TR}, firm::Firm, government::Government, climate::Climate) where {TD, TR}
    aₛ, mₛ, tₛ = committedinitialstate(τᶜ, firm, climate)
    m̄ = τᶜ.ub[2]

    J₁ = aₛ > firm.a₀ ? # Initial jump in abatement
        government.r * (C(aₛ, firm) - C(firm.a₀, firm)) :
        stallingcosts(τᶜ, firm.a₀, mₛ, firm, government, climate)

    if mₛ ≈ m̄ || aₛ ≈ firm.e₀ # Trivial starting point
        taxₛ = τᶜ(SA.SVector(aₛ, mₛ))
        J₃ = tailcosts(aₛ, mₛ, tₛ, taxₛ, firm, government, climate)

        return J₁ + J₃
    end

    terminalboundaryobj = committedboundaryobj(firm.e₀, m̄, τᶜ, firm)

    ā = if terminalboundaryobj ≥ 0
        firm.e₀
    else
        initialterminalboundaryobj = committedboundaryobj(aₛ, m̄, τᶜ, firm)

        if (initialterminalboundaryobj < 0) return convert(TR, Inf) end

        obj = @closure a -> committedboundaryobj(a, m̄, τᶜ, firm)
        brent(obj, (aₛ, firm.e₀))
    end

    t̄, m̄boundary, J₂ = transitioncosts(τᶜ, aₛ, mₛ, tₛ, ā, firm, government, climate)

    τᶜ̄ = if ā < firm.e₀
        firm.r * c(ā, firm)
    else
        x̄ = clamp.(SA.SVector(ā, m̄boundary), τᶜ.lb, τᶜ.ub)
        τᶜ(x̄)
    end
    J₃ = tailcosts(ā, m̄boundary, t̄, τᶜ̄, firm, government, climate)

    return J₁ + J₂ + J₃
end
