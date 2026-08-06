"Tax along the investment boundary"
function taxalongoptimal(a, firm::Firm)
    firm.r * c(a, firm)
end

"Payoff along the investment boundary"
function ω(a, m, firm::Firm, government::Government, climate::Climate)
    government.y₀ * d(m, climate) + l(taxalongoptimal(a, firm), government)
end
function ω(x::AbstractArray, parameters)
    firm, government, climate = parameters

    return ω(x[1], x[2], firm, government, climate)
end

"Computes value, gradient and hessian of 'ω'"
function ωderivatives(a, m, firm::Firm, government::Government, climate::Climate)
    x = SA.MVector(a, m)
    f = Base.Fix2(ω, (firm, government, climate))

    cfg = ForwardDiff.HessianConfig(f, x)
    output = DiffResults.HessianResult(x)

    return ForwardDiff.hessian!(output, f, x, cfg)
end

function singulararccomponents(a, m, firm::Firm, government::Government, climate::Climate)
    derivatives = ωderivatives(a, m, firm, government, climate)

    ∂ₐω, ∂ₘω = DiffResults.gradient(derivatives)
    ∂ₐₐω, ∂ₐₘω, _, _ = DiffResults.hessian(derivatives)

    r = government.r
    curvature = ∂ₐₐω + r * c′(firm)
    den = r * (∂ₐω + r * c(a, firm)) - ∂ₘω - e(a, firm) * ∂ₐₘω

    return curvature, den
end

function ∂ₐM(a, m, firm::Firm, government::Government, climate::Climate)
    curvature, den = singulararccomponents(a, m, firm, government, climate)

    return e(a, firm) * curvature / den
end

function investmentdrift(a, m, firm::Firm, government::Government, climate::Climate)
    curvature, den = singulararccomponents(a, m, firm, government, climate)

    return den / curvature
end

function singularity∂ₐM(a, m, firm::Firm, government::Government, climate::Climate)
    _, den = singulararccomponents(a, m, firm, government, climate)

    return den
end

function gaussianintegral(m, α, β)
    exp(-β * m^2) * SpecialFunctions.erfcx(√β * m + α / (2√β))
end

# Initial payoff
function J₁(aₛ, mₛ, firm::Firm, government::Government, climate::Climate)
    α = government.r / e(firm.a₀, firm)
    β = climate.γ * climate.ζ^2 / 2
    Δm = mₛ - climate.m₀
    discount = exp(-α * Δm)
    gaussianweight = α * √(π / β) / 2

    damagecost = government.y₀ * (
        -expm1(-α * Δm) - gaussianweight * (
            gaussianintegral(climate.m₀, α, β) -
            discount * gaussianintegral(mₛ, α, β)
        )
    )

    jumpcost = government.r * discount * (C(aₛ, firm) - C(firm.a₀, firm))

    return damagecost + jumpcost
end

# Terminal payoff
function J₃(ā, m̄, t̄, firm::Firm, government::Government, climate::Climate)
    discount = exp(-government.r * t̄)
    emissions = e(ā, firm)

    if iszero(emissions)
        return discount * government.y₀ * d(m̄, climate)
    end

    α = government.r / emissions
    β = climate.γ * climate.ζ^2 / 2
    gaussianweight = α * √(π / β) / 2
    damagecost = government.y₀ * (
        1 - gaussianweight * gaussianintegral(m̄, α, β)
    )

    return discount * damagecost
end

# Path payoff in calendar time
function pathdrift(x, parameters, t)
    firm, government, climate, _ = parameters

    @unpack r = government

    a, m, _ = x

    da = investmentdrift(a, m, firm, government, climate)
    dm = e(a, firm)
    dJ = r * exp(-r * t) * (
        ω(a, m, firm, government, climate) + c(a, firm) * da
    )

    return SA.SVector(da, dm, dJ)
end

function singularitycondition(x, _, integrator)
    firm, government, climate, _ = integrator.p
    a, m, _ = x

    return singularity∂ₐM(a, m, firm, government, climate)
end

function endpointcondition(x, _, integrator)
    _, _, _, ā = integrator.p
    a = x[1]

    return ā - a
end

function netzerocondition(x, _, integrator)
    firm, _, _, _ = integrator.p
    a = x[1]

    return e(a, firm)
end

const rosenberck = ODERosenbrock.Rosenbrock23()
const singularitycallback = SciMLBase.ContinuousCallback(singularitycondition, SciMLBase.terminate!)
const endpointcallback = SciMLBase.ContinuousCallback(endpointcondition, SciMLBase.terminate!)
const netzerocallback = SciMLBase.ContinuousCallback(netzerocondition, SciMLBase.terminate!)
const pathcallback = SciMLBase.CallbackSet(singularitycallback, endpointcallback, netzerocallback)

function J(mₛ::T, aₛ, ā, firm::Firm, government::Government, climate::Climate) where T <: Real

    initialcost = J₁(aₛ, mₛ, firm, government, climate)
    tₛ = (mₛ - climate.m₀) / e(firm.a₀, firm)

    if aₛ ≈ ā
        return initialcost + J₃(aₛ, mₛ, tₛ, firm, government, climate)
    end

    parameters = (firm, government, climate, ā)
    x₀ = SA.SVector(aₛ, mₛ, 0)
    prob = SciMLBase.ODEProblem{false}(pathdrift, x₀, (tₛ, Inf), parameters)
    solution = ODE.solve( prob, rosenberck; callback = pathcallback, save_everystep = false, save_start = false)

    t̄ = last(solution.t)
    ā, m̄, pathcost = last(solution.u)
    terminalcost = J₃(ā, m̄, t̄, firm, government, climate)

    return initialcost + pathcost + terminalcost
end
