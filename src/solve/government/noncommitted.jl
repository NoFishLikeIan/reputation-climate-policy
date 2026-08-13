## State space
struct NonCommittedGrid{Tᵠ, Tₘ, Tₐ}
    φgrid::Tᵠ
    mgrid::Tₘ
    agrid::Tₐ
end

function Base.size(grid::NonCommittedGrid)
    (length(grid.φgrid), length(grid.mgrid), length(grid.agrid))
end
function Base.length(grid::NonCommittedGrid)
    prod(size(grid))
end

abstract type NonCommittedTaxMethod end

"Tax condition from an instantaneous hidden-action deviation"
struct OneShotTax <: NonCommittedTaxMethod end

"Tax condition obtained by controlling the complete posterior generator"
struct FullGeneratorTax{T} <: NonCommittedTaxMethod
    upperbound::T
end

struct NonCommittedScalingParameters{T}
    q::T
    W::T
    function NonCommittedScalingParameters(firm::Firm{T}, government::Government{T}) where T
        q = firm.r * c(firm.e₀, firm)
        W = government.y₀
    
        return new{T}(q, W)
    end
end

struct NonCommittedParameters{TC, T, F, G, S, C, TG, TS, TM <: NonCommittedTaxMethod}
    τᶜ::TC
    horizon::T
    firm::F
    government::G
    signal::S
    climate::C
    grid::TG
    scaling::TS
    taxmethod::TM
end

struct CommittedTaxPath{TI, T}
    active::TI
    activeterminal::T
    terminal::T
    tailtax::T
    taildecay::T

    function CommittedTaxPath(active::TI, activeterminal::T, terminal, terminalabatement, firm::Firm, government::Government) where {TI, T}
        tailtax = committedtailtax(zero(T), terminalabatement, firm, government)
        taildecay = firm.r - government.r

        return new{TI, T}(active, activeterminal, terminal, tailtax, taildecay)
    end
end
function (path::CommittedTaxPath{TI, T})(t) where {TI, T}
    if t < path.activeterminal
        return path.active(t)
    elseif path.activeterminal ≤ t ≤ path.terminal
        decay = exp( -path.taildecay * (t - path.activeterminal))
        return path.tailtax * decay
    else
        return zero(T)
    end
end


function committedtaxterminal(activeterminal::T, terminalabatement, firm::Firm, government::Government; tolerance = 0.1taxfactor) where T
    tailtax = committedtailtax(zero(T), terminalabatement, firm, government)

    if tailtax ≤ tolerance
        return activeterminal
    end

    taildecay = firm.r - government.r

    return activeterminal + log(tailtax / tolerance) / taildecay
end

function NonCommittedParameters(
    τᶜ, horizon, grid::NonCommittedGrid,
    firm::Firm, government::Government, signal::Signal, climate::Climate;
    taxmethod::NonCommittedTaxMethod = OneShotTax()
)
    scaling = NonCommittedScalingParameters(firm, government)

    return NonCommittedParameters(
        τᶜ, horizon, firm, government, signal, climate, grid, scaling,
        taxmethod
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

function noncommittedtaxcoefficient(
    ::OneShotTax, ∂ᵩW, _, φ, signal::Signal
)
    -φ * (1 - φ) * (signal.ϵ / signal.σ)^2 * ∂ᵩW
end

function noncommittedtaxcoefficient(
    ::FullGeneratorTax, ∂ᵩW, ∂ᵩᵩW, φ, signal::Signal
)
    return (
        φ^2 * (1 - φ) * (signal.ϵ / signal.σ)^2 *
        (-2∂ᵩW + (1 - φ) * ∂ᵩᵩW)
    )
end

"Minimum-tax solution of the hidden-action condition"
function noncommittedtax(
    taxmethod::OneShotTax, ∂ᵩW, ∂ᵩᵩW, φ, τᶜ,
    signal::Signal, government::Government
)
    taxcoefficient = noncommittedtaxcoefficient(
        taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, signal
    )

    if taxcoefficient <= 0 || τᶜ <= 0
        return zero(taxcoefficient + τᶜ)
    end

    return taxcoefficient * τᶜ / (
        government.r * government.δ + taxcoefficient
    )
end

function fullgeneratorhamiltonian(
    τ, taxcoefficient, τᶜ, government::Government
)
    return (
        government.r * government.δ * τ^2 / 2 +
        taxcoefficient * (τᶜ - τ)^2 / 2
    )
end

"Global solution of the complete-generator tax problem"
function noncommittedtax(
    taxmethod::FullGeneratorTax, ∂ᵩW, ∂ᵩᵩW, φ, τᶜ,
    signal::Signal, government::Government
)
    taxcoefficient = noncommittedtaxcoefficient(
        taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, signal
    )
    curvature = government.r * government.δ + taxcoefficient
    zerotax = zero(taxcoefficient + τᶜ + taxmethod.upperbound)
    upperbound = zerotax + taxmethod.upperbound

    if curvature > 0
        τ = taxcoefficient * τᶜ / curvature

        return clamp(τ, zerotax, upperbound)
    end

    zerovalue = fullgeneratorhamiltonian(
        zerotax, taxcoefficient, τᶜ, government
    )
    uppervalue = fullgeneratorhamiltonian(
        upperbound, taxcoefficient, τᶜ, government
    )

    return uppervalue < zerovalue ? upperbound : zerotax
end

function noncommittedtax(∂ᵩW, φ, τᶜ, signal::Signal, government::Government)
    noncommittedtax(
        OneShotTax(), ∂ᵩW, zero(∂ᵩW), φ, τᶜ, signal, government
    )
end

function noncommittedtaxresidual(
    τ, taxmethod::NonCommittedTaxMethod, ∂ᵩW, ∂ᵩᵩW, φ, τᶜ,
    signal::Signal, government::Government
)
    taxcoefficient = noncommittedtaxcoefficient(
        taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, signal
    )

    return (
        government.r * government.δ * τ -
        taxcoefficient * (τᶜ - τ)
    )
end


function noncommittedtaxviolation(τ, residual, ::OneShotTax)
    iszero(τ) ? max(-residual, zero(residual)) : abs(residual)
end

function noncommittedtaxviolation(τ, residual, method::FullGeneratorTax)
    if iszero(method.upperbound)
        return zero(residual)
    elseif iszero(τ)
        return max(-residual, zero(residual))
    elseif iszero(method.upperbound - τ)
        return max(residual, zero(residual))
    end

    return abs(residual)
end

function noncommittedtaxcomplementarity(τ, residual, ::OneShotTax)
    abs(τ * residual)
end

function noncommittedtaxcomplementarity(τ, residual, method::FullGeneratorTax)
    iszero(method.upperbound) && return zero(residual)

    min(abs(τ * residual), abs((method.upperbound - τ) * residual))
end

function noncommittedtaxgap(τ, _, _, ::OneShotTax, ::Government)
    zero(τ)
end

function noncommittedtaxgap(
    τ, taxcoefficient, τᶜ, method::FullGeneratorTax,
    government::Government
)
    zerotax = zero(τ + taxcoefficient + τᶜ + method.upperbound)
    upperbound = zerotax + method.upperbound
    chosenvalue = fullgeneratorhamiltonian(
        τ, taxcoefficient, τᶜ, government
    )
    minimumvalue = min(
        fullgeneratorhamiltonian(
            zerotax, taxcoefficient, τᶜ, government
        ),
        fullgeneratorhamiltonian(
            upperbound, taxcoefficient, τᶜ, government
        )
    )
    curvature = government.r * government.δ + taxcoefficient

    if curvature > 0
        stationarytax = clamp(
            taxcoefficient * τᶜ / curvature,
            zerotax,
            upperbound
        )
        minimumvalue = min(
            minimumvalue,
            fullgeneratorhamiltonian(
                stationarytax, taxcoefficient, τᶜ, government
            )
        )
    end

    return max(chosenvalue - minimumvalue, zero(chosenvalue))
end

function noncommittedtaxresidual(
    τ, ∂ᵩW, φ, τᶜ, signal::Signal, government::Government
)
    noncommittedtaxresidual(
        τ, OneShotTax(), ∂ᵩW, zero(∂ᵩW), φ, τᶜ, signal, government
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

function noncommittedreversetime(t, parameters::NonCommittedParameters)
    1 - t / parameters.horizon
end

"Coupled firm and government system in reverse time"
function noncommittedreversedrift!(dx, x, parameters::NonCommittedParameters, s)
    @unpack τᶜ, horizon, firm, government, signal, climate, grid, scaling, taxmethod = parameters

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
            # With known non-commitment, q = τ = u = 0 and the stationary
            # damage value is V₃damages(a, m).
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
        τ = noncommittedtax(
            taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, committedtax,
            signal, government
        )
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
    @unpack τᶜ, firm, government, signal, grid, scaling, taxmethod = parameters
    qnormalised, Wnormalised = noncommittedviews(x, grid)

    investment = similar(qnormalised)
    tax = similar(qnormalised)
    expectedtax = similar(qnormalised)
    taxcoefficient = similar(qnormalised)
    taxcurvature = similar(qnormalised)

    calendartime = noncommittedcalendar(s, parameters)
    committedtax = τᶜ(calendartime)

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        φ = grid.φgrid[i]
        a = grid.agrid[k]
        q = scaling.q * qnormalised[i, j, k]
        ∂ᵩW = scaling.W * backwardφderivative(Wnormalised, i, j, k, grid)
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(
            Wnormalised, i, j, k, grid
        )

        taxcoefficient[i, j, k] = noncommittedtaxcoefficient(
            taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, signal
        )
        taxcurvature[i, j, k] = (
            government.r * government.δ + taxcoefficient[i, j, k]
        )
        investment[i, j, k] = noncommittedinvestment(q, a, firm)
        tax[i, j, k] = noncommittedtax(
            taxmethod, ∂ᵩW, ∂ᵩᵩW, φ, committedtax,
            signal, government
        )
        expectedtax[i, j, k] = noncommittedexpectedtax(
            φ, tax[i, j, k], committedtax
        )
    end

    return (;
        investment, tax, expectedtax, taxcoefficient, taxcurvature
    )
end

function noncommittedpoliciesattime(
    solution, parameters::NonCommittedParameters, t
)
    s = noncommittedreversetime(t, parameters)

    return noncommittedpolicies(solution(s), parameters, s)
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
    @unpack τᶜ, firm, government, signal, grid, scaling, taxmethod = parameters
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
        ∂ᵩᵩW = scaling.W * centralφsecondderivative(
            Wnormalised, i, j, k, grid
        )

        investmentgap[i, j, k] = firminvestmentgap(
            q, a, policies.investment[i, j, k], firm
        )
        taxresidual[i, j, k] = noncommittedtaxresidual(
            policies.tax[i, j, k], taxmethod, ∂ᵩW, ∂ᵩᵩW,
            φ, committedtax, signal, government
        )
    end

    return (; investmentgap, taxresidual)
end

function noncommittedkktdiagnostics(x, parameters::NonCommittedParameters, s)
    @unpack firm, grid, taxmethod = parameters
    policies = noncommittedpolicies(x, parameters, s)
    residuals = noncommittedresiduals(x, parameters, s)

    firmviolation = zero(eltype(x))
    taxviolation = zero(eltype(x))
    complementarity = zero(eltype(x))
    taxhamiltoniangap = zero(eltype(x))
    committedtax = parameters.τᶜ(noncommittedcalendar(s, parameters))

    @inbounds for k in eachindex(grid.agrid), j in eachindex(grid.mgrid),
        i in eachindex(grid.φgrid)

        a = grid.agrid[k]
        u = policies.investment[i, j, k]
        τ = policies.tax[i, j, k]
        investmentgap = residuals.investmentgap[i, j, k]
        taxresidual = residuals.taxresidual[i, j, k]
        taxcoefficient = policies.taxcoefficient[i, j, k]

        if !iszero(e(a, firm))
            firmviolation = max(firmviolation, investmentgap)
            complementarity = max(
                complementarity, abs(u * investmentgap)
            )
        end

        taxviolation = max(
            taxviolation,
            noncommittedtaxviolation(τ, taxresidual, taxmethod)
        )
        complementarity = max(
            complementarity,
            noncommittedtaxcomplementarity(τ, taxresidual, taxmethod)
        )
        taxhamiltoniangap = max(
            taxhamiltoniangap,
            noncommittedtaxgap(
                τ, taxcoefficient, committedtax, taxmethod,
                parameters.government
            )
        )
    end

    return (;
        firmviolation, taxviolation, complementarity, taxhamiltoniangap
    )
end
