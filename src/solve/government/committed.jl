## Cost of initial waiting period
function gaussianintegral(m, α, β)
    exp(-β * m^2) * SpecialFunctions.erfcx(√β * m + α / (2√β))
end

"Annualised cost before investment begins"
function J₁(mₛ, firm::Firm, government::Government, climate::Climate)
    α = government.r / e(firm.a₀, firm)
    β = climate.γ * climate.ζ^2 / 2
    Δm = mₛ - climate.m₀
    discount = exp(-α * Δm)
    gaussianweight = α * √(π / β) / 2

    return government.y₀ * (
        -expm1(-α * Δm) - gaussianweight * (
            gaussianintegral(climate.m₀, α, β) -
            discount * gaussianintegral(mₛ, α, β)
        )
    )
end

## Cost at transition end
"Current terminal value under permanent partial abatement"
function V₃(ā, m̄, firm::Firm, government::Government, climate::Climate)
    if ā ≈ firm.e₀
        return government.y₀ * d(m̄, climate)
    end

    emissions = e(ā, firm)
    α = government.r / emissions
    β = climate.γ * climate.ζ^2 / 2
    gaussianweight = α * √(π / β) / 2

    return government.y₀ * (1 - gaussianweight * gaussianintegral(m̄, α, β))
end

function ∂ₘV₃(ā, m̄, firm::Firm, government::Government, climate::Climate)
    if ā ≈ firm.e₀
        return government.y₀ * d′(m̄, climate)
    end

    emissions = e(ā, firm)
    α = government.r / emissions

    return α * (V₃(ā, m̄, firm, government, climate) - government.y₀ * d(m̄, climate))
end

## Cost of the transition
# Utility states
"State of the committed planner optimization composed of initial abatemnet time 'tₛ', horizon of abatement efforts 't̄', and terminal abatemnet level 'ā'. The path is determined by local optimality."
struct CommittedState{T} <: SA.FieldVector{3, T}
    tₛ ::T # Initial abatemnet time
    t̄::T # Abatement horizon
    ā::T # Terminal abatement level 
end

struct CommittedParameters{F, G, C}
    firm::F
    government::G
    climate::C
end

struct ScalingParameters{T}
    centre::T
    scale::T
end
function ScalingParameters(parameters::CommittedParameters)
    @unpack firm, government, climate = parameters

    horizon = adjustmenthorizon(firm)
    abatementscope = e(firm.a₀, firm)
    taxscale = firm.r * c(firm.e₀, firm)

    centre = SA.SVector(climate.m₀, firm.a₀, 0., 0., 0., 0., 0.)
    scale = SA.SVector(
        abatementscope * horizon,
        abatementscope,
        abatementscope / horizon,
        government.r * taxscale,
        taxscale,
        government.r * government.δ * firm.ξ * taxscale,
        government.y₀,
    )


    return ScalingParameters(centre, scale)
end

function physicalstate(x, scaling::ScalingParameters)
    @. scaling.centre + scaling.scale * x
end
function normalisedstate(x, scaling::ScalingParameters)
    @. (x - scaling.centre) / scaling.scale
end
function physicalpayoff(P, scaling::ScalingParameters)
    scaling.centre[7] + P * scaling.scale[7]
end

struct CommittedPathParameters{TS <: CommittedState, TP <: CommittedParameters, S <: ScalingParameters}
    y::TS
    parameters::TP
    scaling::S
end
function CommittedPathParameters(tₛ, duration, ā, firm::Firm, government::Government, climate::Climate)
    x = CommittedState(tₛ, duration, ā)
    parameters = CommittedParameters(firm, government, climate)
    scaling = ScalingParameters(parameters)

    return CommittedPathParameters(x, parameters, scaling)
end

function committedtax(λᵤ, firm::Firm, government::Government)
    λᵤ / (government.r * government.δ * firm.ξ)
end

function committedacceleration(a, u, τ, firm::Firm)
    firm.r * u + (firm.r * c(a, firm) - τ) / firm.ξ
end

function committedflowcost(a, m, u, τ, firm::Firm, government::Government, climate::Climate)
    government.y₀ * d(m, climate) + l(τ, government) + investmentcost(a, u, firm)
end

function committedhamiltonian(x, firm::Firm, government::Government, climate::Climate)
    m, a, u, λₘ, λₐ, λᵤ, _ = x
    τ = committedtax(λᵤ, firm, government)
    v = committedacceleration(a, u, τ, firm)
    flowcost = committedflowcost(a, m, u, τ, firm, government, climate)

    return government.r * flowcost + λₘ * e(a, firm) + λₐ * u + λᵤ * v
end

"Canonical system in calendar time"
function committeddrift(x, parameters, _)
    @unpack firm, government, climate = parameters
    m, a, u, λₘ, λₐ, λᵤ, P = x

    τ = committedtax(λᵤ, firm, government)
    v = committedacceleration(a, u, τ, firm)
    flowcost = committedflowcost(a, m, u, τ, firm, government, climate)

    dm = e(a, firm)
    da = u
    du = v
    dλₘ = government.r * (λₘ - government.y₀ * d′(m, climate))
    dλₐ = (
        government.r * λₐ - government.r * c′(a, firm) * u + λₘ -
        firm.r * c′(a, firm) * λᵤ / firm.ξ
    )
    dλᵤ = (
        (government.r - firm.r) * λᵤ -
        government.r * (c(a, firm) + firm.ξ * u) - λₐ
    )
    dP = government.r * (P - flowcost)

    return SA.SVector(dm, da, du, dλₘ, dλₐ, dλᵤ, dP)
end

"Canonical system on the normalised active interval"
function committednormaliseddrift(x, p::CommittedPathParameters, s)
    @unpack y, parameters, scaling = p
    physical = physicalstate(x, scaling)
    drift = committeddrift(physical, parameters, s)

    return y.t̄ .* drift ./ scaling.scale
end
function committednormaliseddrift!(dx, x, p, s)
    dx .= committednormaliseddrift(x, p, s)
end

function initialcondition!(res, x, p::CommittedPathParameters)
    @unpack y, parameters, scaling = p
    @unpack firm, government, climate = parameters

    mₛ = climate.m₀ + e(firm.a₀, firm) * y.tₛ

    res[1] = x[1] - (mₛ - scaling.centre[1]) / scaling.scale[1]
    res[2] = x[2] - (firm.a₀ - scaling.centre[2]) / scaling.scale[2]
    res[3] = x[3] + (scaling.centre[3] / scaling.scale[3])

    return
end
function terminalcondition!(res, x, p::CommittedPathParameters)
    @unpack y, parameters, scaling = p
    @unpack firm, government, climate = parameters

    physical = physicalstate(x, scaling)
    m̄, ā, ū, λₘ, _, _, P = physical
    terminalλₘ = ∂ₘV₃(y.ā, m̄, firm, government, climate)

    
    res[1] = (ā - y.ā) / scaling.scale[2]
    res[2] = ū / scaling.scale[3]
    res[3] = (λₘ - terminalλₘ) / scaling.scale[4]
    res[4] = (P - V₃(y.ā, m̄, firm, government, climate)) / scaling.scale[7]

    return
end

function committedinitialguess(s, p::CommittedPathParameters)
    @unpack y, parameters, scaling = p
    @unpack firm, government, climate = parameters

    progress = s^2 * (3 - 2s)
    Δa = y.ā - firm.a₀

    mₛ = climate.m₀ + e(firm.a₀, firm) * y.tₛ
    m = mₛ + y.t̄ * (e(firm.a₀, firm) * s - Δa * (s^3 - s^4 / 2))

    a = firm.a₀ + Δa * progress
    u = 6Δa * s * (1 - s) / y.t̄
    m̄ = mₛ + y.t̄ * (e(firm.a₀, firm) - Δa / 2)

    λₘ = ∂ₘV₃(y.ā, m̄, firm, government, climate)
    λₐ = zero(λₘ)
    
    τ = firm.r * c(y.ā, firm)
    λᵤ = government.r * government.δ * firm.ξ * τ
    P = V₃(y.ā, m̄, firm, government, climate)
    physical = SA.MVector(m, a, u, λₘ, λₐ, λᵤ, P)

    return normalisedstate(physical, p.scaling)
end

function solvecommittedpath(pathparameters::CommittedPathParameters; fallbackdt = 1e-2)
    x0 = committedinitialguess(0., pathparameters)

    problem = BVP.TwoPointBVProblem{true}(
        committednormaliseddrift!,
        (initialcondition!, terminalcondition!),
        x0,
        (0., 1.),
        pathparameters;
        bcresid_prototype = (zeros(SA.MVector{3}), zeros(SA.MVector{4}))
    )

    solution = BVP.solve(problem, BVP.Shooting(ODE.Tsit5()); save_everystep = false)

    if !SciMLBase.successful_retcode(solution)
        solution = BVP.solve(problem, BVP.MIRK4(); dt = fallbackdt, save_everystep = false)
    end
    
    return solution.u[1]
end

function committedvalue(solution, pathparameters::CommittedPathParameters)
    @unpack y, parameters, scaling = pathparameters
    @unpack firm, government, climate = parameters

    mₛ = climate.m₀ + e(firm.a₀, firm) * y.tₛ
    V = physicalpayoff(solution[7], scaling)

    return J₁(mₛ, firm, government, climate) + exp(-government.r * y.tₛ) * V
end

function committedobjective(y, objparameters)
    committedobjective(CommittedState(y[1], y[2], y[3]), objparameters)
end
function committedobjective(y::CommittedState{T}, (parameters, scaling)) where T
    pathparameters = CommittedPathParameters(y, parameters, scaling)

    solution = solvecommittedpath(pathparameters)

    return committedvalue(solution, pathparameters)
end

function committedpath(solution)
    parameters = solution.prob.p
    time = parameters.tₛ .+ parameters.duration .* solution.t
    states = map(x -> physicalstate(x, parameters), solution.u)

    return time, states
end

function committedpathdiagnostics(solution)
    parameters = solution.prob.p
    firm = parameters.firm
    government = parameters.government
    _, states = committedpath(solution)
    taxes = map(x -> committedtax(x[6], firm, government), states)
    terminal = last(states)
    m̄, ā = terminal[1:2]
    terminalvalue = iszero(e(parameters.ā, firm)) ?
        V₃(m̄, government, parameters.climate) :
        V₃(parameters.ā, m̄, firm, government, parameters.climate)

    return (
        minimumabatement = minimum(x -> x[2], states),
        maximumabatement = maximum(x -> x[2], states),
        minimuminvestment = minimum(x -> x[3], states),
        minimumtax = minimum(taxes),
        maximumtax = maximum(taxes),
        terminaltimecondition = committedhamiltonian(
            terminal, firm, government, parameters.climate
        ) - government.r * terminalvalue,
    )
end
