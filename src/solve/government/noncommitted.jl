## State space
struct NonCommittedGrid{Tᵠ, Tₘ, Tₐ}
    φgrid::Tᵠ
    mgrid::Tₘ
    agrid::Tₐ
end

Base.size(grid::NonCommittedGrid) = (
    length(grid.φgrid), length(grid.mgrid), length(grid.agrid)
)
Base.length(grid::NonCommittedGrid) = prod(size(grid))

struct NonCommittedScalingParameters{T}
    q::T
    W::T
end
function NonCommittedScalingParameters(firm::Firm, government::Government)
    q = firm.r * c(firm.e₀, firm)
    W = government.y₀

    return NonCommittedScalingParameters(q, W)
end

struct NonCommittedParameters{TC, T, F, G, S, C, TG, TS}
    τᶜ::TC
    horizon::T
    firm::F
    government::G
    signal::S
    climate::C
    grid::TG
    scaling::TS
end

struct CommittedTaxPath{TI, T}
    active::TI
    activeterminal::T
    terminal::T
    tailtax::T
    taildecay::T
end
function (path::CommittedTaxPath)(t)
    if t < path.activeterminal
        return path.active(t)
    elseif t < path.terminal
        return path.tailtax * exp(
            -path.taildecay * (t - path.activeterminal)
        )
    end

    return zero(path.tailtax)
end

function CommittedTaxPath(
    active, activeterminal, terminal, terminalabatement,
    firm::Firm, government::Government
)
    tailtax = committedtailtax(
        zero(activeterminal), terminalabatement, firm, government
    )
    taildecay = firm.r - government.r

    return CommittedTaxPath(
        active, activeterminal, terminal, tailtax, taildecay
    )
end

function committedtaxterminal(
    activeterminal, terminalabatement,
    firm::Firm, government::Government;
    tolerance = 0.1taxfactor
)
    tailtax = committedtailtax(
        zero(activeterminal), terminalabatement, firm, government
    )

    if tailtax <= tolerance
        return activeterminal
    end

    taildecay = firm.r - government.r
    taildecay > 0 || throw(ArgumentError(
        "The committed tax tail requires r_f > r_g."
    ))

    return activeterminal + log(tailtax / tolerance) / taildecay
end

function NonCommittedParameters(
    τᶜ, horizon, grid::NonCommittedGrid,
    firm::Firm, government::Government, signal::Signal, climate::Climate
)
    scaling = NonCommittedScalingParameters(firm, government)

    return NonCommittedParameters(
        τᶜ, horizon, firm, government, signal, climate, grid, scaling
    )
end

function noncommittedviews(x, grid::NonCommittedGrid)
    n = length(grid)
    q = reshape(view(x, 1:n), size(grid))
    W = reshape(view(x, (n + 1):(2n)), size(grid))

    return q, W
end

## Policies
function noncommittedexpectedtax(φ, τ, τᶜ)
    φ * τᶜ + (1 - φ) * τ
end

function noncommittedinvestment(q, a, firm::Firm)
    if iszero(e(a, firm))
        return zero(q)
    end

    return max((q / firm.r - c(a, firm)) / firm.ξ, zero(q))
end

"Minimum-tax solution of the non-committed government's one-shot condition"
function noncommittedtax(∂ᵩW, φ, τᶜ, signal::Signal, government::Government)
    reputationvalue = -φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * ∂ᵩW

    if reputationvalue <= 0 || τᶜ <= 0
        return zero(reputationvalue + τᶜ)
    end

    return reputationvalue * τᶜ / (
        government.r * government.δ + reputationvalue
    )
end

function noncommittedtaxresidual(τ, ∂ᵩW, φ, τᶜ, signal::Signal, government::Government)
    reputationcoefficient = φ * (1 - φ) * (signal.ϵ / signal.σ)^2

    return (
        government.r * government.δ * τ +
        reputationcoefficient * (τᶜ - τ) * ∂ᵩW
    )
end

function firminvestmentgap(q, a, u, firm::Firm)
    q - firm.r * (c(a, firm) + firm.ξ * u)
end

function noncommittedflowcost(
    a, m, u, τ, firm::Firm, government::Government, climate::Climate
)
    government.y₀ * d(m, climate) + l(τ, government) + investmentcost(a, u, firm)
end

## Finite differences
@inline function forwardmderivative(x, i, j, k, grid::NonCommittedGrid)
    # The upper boundary should be placed beyond the states reachable before T.
    if j == length(grid.mgrid)
        return zero(eltype(x))
    end

    return (x[i, j + 1, k] - x[i, j, k]) / (
        grid.mgrid[j + 1] - grid.mgrid[j]
    )
end

@inline function forwardaderivative(x, i, j, k, grid::NonCommittedGrid)
    if k == length(grid.agrid)
        return zero(eltype(x))
    end

    return (x[i, j, k + 1] - x[i, j, k]) / (
        grid.agrid[k + 1] - grid.agrid[k]
    )
end

@inline function backwardφderivative(x, i, j, k, grid::NonCommittedGrid)
    if i == 1
        return zero(eltype(x))
    end

    return (x[i, j, k] - x[i - 1, j, k]) / (
        grid.φgrid[i] - grid.φgrid[i - 1]
    )
end

@inline function centralφsecondderivative(x, i, j, k, grid::NonCommittedGrid)
    if i == 1 || i == length(grid.φgrid)
        return zero(eltype(x))
    end

    leftstep = grid.φgrid[i] - grid.φgrid[i - 1]
    rightstep = grid.φgrid[i + 1] - grid.φgrid[i]
    leftderivative = (x[i, j, k] - x[i - 1, j, k]) / leftstep
    rightderivative = (x[i + 1, j, k] - x[i, j, k]) / rightstep

    return 2 * (rightderivative - leftderivative) / (leftstep + rightstep)
end

## Terminal values
"Terminal state after the committed tax tail has become negligible"
function noncommittedterminalstate(parameters::NonCommittedParameters)
    @unpack firm, government, climate, grid, scaling = parameters

    T = promote_type(
        eltype(grid.φgrid), eltype(grid.mgrid), eltype(grid.agrid),
        typeof(firm.r), typeof(government.r)
    )
    x = zeros(T, 2length(grid))
    _, W = noncommittedviews(x, grid)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        a = grid.agrid[k]
        m = grid.mgrid[j]
        W[i, j, k] = V₃damages(a, m, firm, government, climate) / scaling.W
    end

    return x
end

## Coupled firm-government system
function noncommittedcalendar(s, parameters::NonCommittedParameters)
    (1 - s) * parameters.horizon
end

"Coupled firm and government system in reverse time"
function noncommittedreversedrift!(dx, x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, horizon, firm, government, signal, climate, grid, scaling = parameters

    qnormalised, Wnormalised = noncommittedviews(x, grid)
    dqnormalised, dWnormalised = noncommittedviews(dx, grid)

    calendartime = noncommittedcalendar(s, parameters)
    committedtax = τᶜ(calendartime)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        φ = grid.φgrid[i]
        m = grid.mgrid[j]
        a = grid.agrid[k]

        if iszero(φ)
            dqnormalised[i, j, k] = zero(eltype(dx))
            dWnormalised[i, j, k] = zero(eltype(dx))
            continue
        end

        q = scaling.q * qnormalised[i, j, k]
        W = scaling.W * Wnormalised[i, j, k]

        ∂ₘq = scaling.q * forwardmderivative(qnormalised, i, j, k, grid)
        ∂ₐq = scaling.q * forwardaderivative(qnormalised, i, j, k, grid)
        ∂ᵩᵩq = scaling.q * centralφsecondderivative(qnormalised, i, j, k, grid)

        ∂ₘW = scaling.W * forwardmderivative(Wnormalised, i, j, k, grid)
        ∂ₐW = scaling.W * forwardaderivative(Wnormalised, i, j, k, grid)
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(Wnormalised, i, j, k, grid)

        u = noncommittedinvestment(q, a, firm)
        τ = noncommittedtax(∂ᵩW, φ, committedtax, signal, government)
        τᵉ = noncommittedexpectedtax(φ, τ, committedtax)

        signaltonoise = χ(τ, committedtax, signal)
        bᵩ = beliefdrift(signaltonoise, φ)
        σᵩ = beliefdiffusion(signaltonoise, φ)

        dq = (
            -firm.r * q + firm.r * (τᵉ - c′(a, firm) * u) +
            e(a, firm) * ∂ₘq + u * ∂ₐq + σᵩ^2 * ∂ᵩᵩq / 2
        )
        flowcost = noncommittedflowcost(
            a, m, u, τ, firm, government, climate
        )
        dW = (
            -government.r * W + government.r * flowcost +
            e(a, firm) * ∂ₘW + u * ∂ₐW +
            bᵩ * ∂ᵩW + σᵩ^2 * ∂ᵩᵩW / 2
        )

        dqnormalised[i, j, k] = horizon * dq / scaling.q
        dWnormalised[i, j, k] = iszero(e(a, firm)) ?
            zero(eltype(dx)) : horizon * dW / scaling.W
    end

    return dx
end

function noncommittedjacobianprototype(grid::NonCommittedGrid)
    nᵠ, nₘ, nₐ = size(grid)
    n = length(grid)
    I = Int[]
    J = Int[]

    sizehint!(I, 28n)
    sizehint!(J, 28n)

    @inbounds for k in 1:nₐ, j in 1:nₘ, i in 1:nᵠ
        node = i + (j - 1) * nᵠ + (k - 1) * nᵠ * nₘ
        neighbours = Int[node]

        i > 1 && push!(neighbours, node - 1)
        i < nᵠ && push!(neighbours, node + 1)
        j > 1 && push!(neighbours, node - nᵠ)
        j < nₘ && push!(neighbours, node + nᵠ)
        k > 1 && push!(neighbours, node - nᵠ * nₘ)
        k < nₐ && push!(neighbours, node + nᵠ * nₘ)

        for row in (node, n + node), column in neighbours
            push!(I, row)
            push!(J, column)
            push!(I, row)
            push!(J, n + column)
        end
    end

    T = promote_type(
        eltype(grid.φgrid), eltype(grid.mgrid), eltype(grid.agrid)
    )

    return SparseArrays.sparse(I, J, ones(T, length(I)), 2n, 2n)
end

function noncommittedproblem(parameters::NonCommittedParameters)
    x₀ = noncommittedterminalstate(parameters)
    jacobian = noncommittedjacobianprototype(parameters.grid)
    drift = SciMLBase.ODEFunction(
        noncommittedreversedrift!; jac_prototype = jacobian
    )

    return SciMLBase.ODEProblem(drift, x₀, (0., 1.), parameters)
end

function solvenoncommitted(
    parameters::NonCommittedParameters;
    algorithm = BDF.QNDF(autodiff = ADTypes.AutoFiniteDiff()),
    saveat = range(0., 1.; length = 101),
    kwargs...
)
    problem = noncommittedproblem(parameters)

    return ODE.solve(problem, algorithm; saveat, kwargs...)
end

## Diagnostics
function noncommittedpolicies(x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, firm, government, signal, grid, scaling = parameters
    qnormalised, Wnormalised = noncommittedviews(x, grid)

    investment = similar(qnormalised)
    tax = similar(qnormalised)
    expectedtax = similar(qnormalised)

    calendartime = noncommittedcalendar(s, parameters)
    committedtax = τᶜ(calendartime)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        φ = grid.φgrid[i]
        a = grid.agrid[k]
        q = scaling.q * qnormalised[i, j, k]
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)

        investment[i, j, k] = noncommittedinvestment(q, a, firm)
        tax[i, j, k] = noncommittedtax(
            ∂ᵩW, φ, committedtax, signal, government
        )
        expectedtax[i, j, k] = noncommittedexpectedtax(
            φ, tax[i, j, k], committedtax
        )
    end

    return (; investment, tax, expectedtax)
end

function noncommittedvalues(x, parameters::NonCommittedParameters)
    qnormalised, Wnormalised = noncommittedviews(x, parameters.grid)

    return (
        q = parameters.scaling.q .* qnormalised,
        W = parameters.scaling.W .* Wnormalised,
    )
end

function noncommittedtime(solution, parameters::NonCommittedParameters)
    noncommittedcalendar.(solution.t, Ref(parameters))
end

function noncommittedresiduals(x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, firm, government, signal, grid, scaling = parameters
    qnormalised, Wnormalised = noncommittedviews(x, grid)
    policies = noncommittedpolicies(x, parameters, s)

    investmentgap = similar(qnormalised)
    taxresidual = similar(qnormalised)
    committedtax = τᶜ(noncommittedcalendar(s, parameters))

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        φ = grid.φgrid[i]
        a = grid.agrid[k]
        q = scaling.q * qnormalised[i, j, k]
        ∂ᵩW = scaling.W * backwardφderivative(
            Wnormalised, i, j, k, grid
        )

        investmentgap[i, j, k] = firminvestmentgap(
            q, a, policies.investment[i, j, k], firm
        )
        taxresidual[i, j, k] = noncommittedtaxresidual(
            policies.tax[i, j, k], ∂ᵩW, φ, committedtax,
            signal, government
        )
    end

    return (; investmentgap, taxresidual)
end

function noncommittedkktdiagnostics(x, parameters::NonCommittedParameters, s)
    @unpack firm, grid = parameters
    policies = noncommittedpolicies(x, parameters, s)
    residuals = noncommittedresiduals(x, parameters, s)

    firmviolation = zero(eltype(x))
    taxviolation = zero(eltype(x))
    complementarity = zero(eltype(x))

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        a = grid.agrid[k]
        u = policies.investment[i, j, k]
        τ = policies.tax[i, j, k]
        investmentgap = residuals.investmentgap[i, j, k]
        taxresidual = residuals.taxresidual[i, j, k]

        if !iszero(e(a, firm))
            firmviolation = max(firmviolation, investmentgap)
            complementarity = max(
                complementarity, abs(u * investmentgap)
            )
        end

        taxviolation = max(taxviolation, -taxresidual)
        complementarity = max(
            complementarity, abs(τ * taxresidual)
        )
    end

    return (; firmviolation, taxviolation, complementarity)
end
