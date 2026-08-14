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
"Annualised current damage value under permanent abatement"
function V₃damages(ā, m̄, firm::Firm, government::Government, climate::Climate)
    if iszero(e(ā, firm))
        return government.y₀ * d(m̄, climate)
    end

    emissions = e(ā, firm)
    α = government.r / emissions
    β = climate.γ * climate.ζ^2 / 2
    gaussianweight = α * √(π / β) / 2

    return government.y₀ * (1 - gaussianweight * gaussianintegral(m̄, α, β))
end

"Annualised current cost of the least-cost tax tail implementing partial abatement"
function V₃tax(ā, firm::Firm, government::Government)
    government.r * government.δ * (2firm.r - government.r) * c(ā, firm)^2 / 2
end

"Annualised current terminal value under permanent partial abatement"
function V₃(ā, m̄, firm::Firm, government::Government, climate::Climate)
    value = V₃damages(ā, m̄, firm, government, climate)

    if !iszero(e(ā, firm))
        value += V₃tax(ā, firm, government)
    end

    return value
end

function ∂ₘV₃(ā, m̄, firm::Firm, government::Government, climate::Climate)
    if iszero(e(ā, firm))
        return government.y₀ * d′(m̄, climate)
    end

    emissions = e(ā, firm)
    α = government.r / emissions
    damages = V₃damages(ā, m̄, firm, government, climate)

    return α * (damages - government.y₀ * d(m̄, climate))
end

## Cost of the transition
# Utility states
"State of the committed planner optimisation composed of the abatement horizon 't̄' and terminal abatement level 'ā'."
struct CommittedState{T} <: SA.FieldVector{2, T}
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
function CommittedPathParameters(duration, ā, firm::Firm, government::Government, climate::Climate)
    x = CommittedState(duration, ā)
    parameters = CommittedParameters(firm, government, climate)
    scaling = ScalingParameters(parameters)

    return CommittedPathParameters(x, parameters, scaling)
end

function committedtax(λᵤ, firm::Firm, government::Government)
    λᵤ / (government.r * government.δ * firm.ξ)
end

function committedhamiltonian(x, firm::Firm, government::Government, climate::Climate)
    m, a, u, λₘ, λₐ, λᵤ, _ = x
    τ = committedtax(λᵤ, firm, government)
    v = investmentratedrift(a, u, τ, firm)
    flowcost = transitionflowcost(a, m, u, τ, firm, government, climate)

    return government.r * flowcost +
        λₘ * cumulativeemissionsdrift(a, firm) +
        λₐ * abatementdrift(u) + λᵤ * v
end

"Canonical system in calendar time"
function committeddrift(x, parameters, _)
    @unpack firm, government, climate = parameters
    m, a, u, λₘ, λₐ, λᵤ, P = x

    τ = committedtax(λᵤ, firm, government)
    flowcost = transitionflowcost(a, m, u, τ, firm, government, climate)

    dm = cumulativeemissionsdrift(a, firm)
    da = abatementdrift(u)
    du = investmentratedrift(a, u, τ, firm)
    dλₘ = cumulativeemissionscostatedrift(λₘ, m, government, climate)
    dλₐ = abatementcostatedrift(λₘ, λₐ, λᵤ, a, u, firm, government)
    dλᵤ = investmentratecostatedrift(λₐ, λᵤ, a, u, firm, government)
    dP = annualisedcostdrift(P, flowcost, government)

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
    @unpack parameters, scaling = p
    @unpack firm, climate = parameters
    physical = physicalstate(x, scaling)
    m, a, _, _, _, λᵤ, _ = physical

    res[1] = (m - climate.m₀) / scaling.scale[1]
    res[2] = (a - firm.a₀) / scaling.scale[2]
    res[3] = λᵤ / scaling.scale[6]

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

function committedinitialprofile(s)
    progress = s * (2 - s)
    investmentrate = 2 * (1 - s)
    cumulativeabatement = s^2 - s^3 / 3

    return progress, investmentrate, cumulativeabatement
end

function committedinitialguess(s, p::CommittedPathParameters)
    @unpack y, parameters, scaling = p
    @unpack firm, government, climate = parameters

    progress, investmentrate, cumulativeabatement = committedinitialprofile(s)
    _, _, totalabatement = committedinitialprofile(one(s))
    Δa = y.ā - firm.a₀

    m = climate.m₀ + y.t̄ * (
        cumulativeemissionsdrift(firm.a₀, firm) * s -
        Δa * cumulativeabatement
    )
    
    a = firm.a₀ + Δa * progress
    u = Δa * investmentrate / y.t̄
    m̄ = climate.m₀ + y.t̄ * (
        cumulativeemissionsdrift(firm.a₀, firm) -
        Δa * totalabatement
    )

    λₘ = ∂ₘV₃(y.ā, m̄, firm, government, climate)
    λₐ = zero(λₘ)
    
    τ = firm.r * c(y.ā, firm)
    λ̄ᵤ = government.r * government.δ * firm.ξ * τ
    λᵤ = λ̄ᵤ * progress
    P = V₃(y.ā, m̄, firm, government, climate)
    physical = SA.MVector(m, a, u, λₘ, λₐ, λᵤ, P)

    return normalisedstate(physical, p.scaling)
end

function committedpathproblem(pathparameters::CommittedPathParameters)
    x0 = committedinitialguess(0., pathparameters)

    return BVP.TwoPointBVProblem{true}(
        committednormaliseddrift!,
        (initialcondition!, terminalcondition!),
        x0,
        (0., 1.),
        pathparameters;
        bcresid_prototype = (zeros(SA.MVector{3}), zeros(SA.MVector{4}))
    )
end

function solvecommittedpath(pathparameters::CommittedPathParameters; dt = 1e-2)
    problem = committedpathproblem(pathparameters)

    solution = BVP.solve(problem, BVP.MIRK4(); dt, save_everystep = false)
    
    return solution.u[1]
end

function committedvalue(solution, pathparameters::CommittedPathParameters)
    physicalpayoff(solution[7], pathparameters.scaling)
end

function committedobjective(y, objparameters)
    committedobjective(CommittedState(y[1], y[2]), objparameters)
end
function committedobjective(y::CommittedState, (parameters, scaling))
    pathparameters = CommittedPathParameters(y, parameters, scaling)

    solution = solvecommittedpath(pathparameters)

    return committedvalue(solution, pathparameters)
end

function committedtailpath(
    terminal, y::CommittedState, parameters::CommittedParameters;
    horizon = 100., dt = 0.5
)
    @unpack firm, government = parameters
    m̄ = terminal[1]
    elapsedtime = range(0., horizon; step = dt)

    states = map(elapsedtime) do t
        SA.SVector(
            m̄ + cumulativeemissionsdrift(y.ā, firm) * t,
            y.ā,
            zero(y.ā)
        )
    end
    taxes = map(t -> committedtailtax(t, y.ā, firm, government), elapsedtime)
    time = @. y.t̄ + elapsedtime

    return states, taxes, time
end


function committedpathdiagnostics(
    yopt, parameters::CommittedParameters, scaling
)
    @unpack firm, government, climate = parameters

    y = CommittedState(yopt...)
    pathparameters = CommittedPathParameters(y, parameters, scaling)
    problem = committedpathproblem(pathparameters)

    solutionpath = BVP.solve(problem, BVP.MIRK4(); dt = 1e-2)
    states = [physicalstate(u, scaling) for u in solutionpath.u]

    taxes = map(x -> committedtax(x[6], firm, government), states)
    terminal = last(states)
    m̄, ā = terminal[1:2]

    terminalhamiltonian = committedhamiltonian(
        terminal, firm, government, parameters.climate
    ) - government.r * V₃(y.ā, m̄, firm, government, climate)

    time = @. y.t̄ * solutionpath.t

    return states, taxes, time, terminalhamiltonian
end
