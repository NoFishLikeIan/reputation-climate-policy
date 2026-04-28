"The joint equilibrium unknown, stored as named arrays but usable as a vector by solvers."
struct EquilibriumState{ T, TΨ <: AbstractMatrix{T}, TV <: AbstractArray{T, 3}, TΦ <: AbstractArray{T, 3}, TW <: AbstractMatrix{T}, TΘ <: AbstractMatrix{T}} <: AbstractVector{T}
    continuation::TΨ
    expost::TV
    investment::TΦ
    welfare::TW
    taxes::TΘ
end

"Copy the current `firmvalue` and `welfare` into the structured nonlinear state."
function EquilibriumState(firmvalue::FV, welfare::TW) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    return EquilibriumState(
        copy(firmvalue.continuation.V),
        copy(firmvalue.expost.V),
        copy(firmvalue.expost.P),
        copy(welfare.V),
        copy(welfare.P),
    )
end

Base.IndexStyle(::Type{<:EquilibriumState}) = IndexLinear()
Base.size(x::EquilibriumState) = (length(x),)
Base.length(x::EquilibriumState) = length(x.continuation) + length(x.expost) + length(x.investment) + length(x.welfare) + length(x.taxes)

function Base.getindex(x::EquilibriumState, i::Int)
    @boundscheck checkbounds(x, i)

    n = length(x.continuation)
    if i <= n
        return x.continuation[i]
    end

    i -= n
    n = length(x.expost)
    if i <= n
        return x.expost[i]
    end

    i -= n
    n = length(x.investment)
    if i <= n
        return x.investment[i]
    end

    i -= n
    n = length(x.welfare)
    if i <= n
        return x.welfare[i]
    end

    return x.taxes[i - n]
end

function Base.setindex!(x::EquilibriumState, value, i::Int)
    @boundscheck checkbounds(x, i)

    n = length(x.continuation)
    if i <= n
        x.continuation[i] = value
        return x
    end

    i -= n
    n = length(x.expost)
    if i <= n
        x.expost[i] = value
        return x
    end

    i -= n
    n = length(x.investment)
    if i <= n
        x.investment[i] = value
        return x
    end

    i -= n
    n = length(x.welfare)
    if i <= n
        x.welfare[i] = value
        return x
    end

    x.taxes[i - n] = value
    return x
end

function Base.similar(x::EquilibriumState, ::Type{T}, dims::Dims{1}) where T
    if dims == size(x)
        return EquilibriumState(
            similar(x.continuation, T),
            similar(x.expost, T),
            similar(x.investment, T),
            similar(x.welfare, T),
            similar(x.taxes, T),
        )
    end

    return Array{T}(undef, dims)
end

Base.similar(x::EquilibriumState, ::Type{T}, inds::Tuple{<:AbstractUnitRange}) where T = similar(x, T, (length(first(inds)),))
Base.similar(x::EquilibriumState, ::Type{T}) where T = similar(x, T, size(x))
Base.similar(x::EquilibriumState) = similar(x, eltype(x))

function Base.copyto!(to::EquilibriumState, from::EquilibriumState)
    copyto!(to.continuation, from.continuation)
    copyto!(to.expost, from.expost)
    copyto!(to.investment, from.investment)
    copyto!(to.welfare, from.welfare)
    copyto!(to.taxes, from.taxes)

    return to
end

function Base.copy(x::EquilibriumState)
    y = similar(x)
    copyto!(y, x)

    return y
end

function Base.fill!(x::EquilibriumState, value)
    fill!(x.continuation, value)
    fill!(x.expost, value)
    fill!(x.investment, value)
    fill!(x.welfare, value)
    fill!(x.taxes, value)

    return x
end

function Base.zero(x::EquilibriumState)
    y = similar(x)
    fill!(y, zero(eltype(x)))

    return y
end

function firmvaluefromstate(state::EquilibriumState)
    continuation = ValueFunction(state.continuation, similar(state.continuation))
    policy = clamp.(state.investment, 0, 1)
    expost = ValueFunction(state.expost, policy)

    return FirmValue(continuation, expost)
end

function welfarefromstate(state::EquilibriumState, τᶜ)
    policy = clamp.(state.taxes, 0, 2τᶜ)

    return ValueFunction(state.welfare, policy)
end

"Copy the structured nonlinear state back into `firmvalue` and `welfare`."
function unpackequilibrium!(firmvalue::FV, welfare::TW, state::EquilibriumState; φlims = (0., Inf), τlims = (0., Inf)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    copyto!(firmvalue.continuation.V, state.continuation)
    copyto!(firmvalue.expost.V, state.expost)
    copyto!(firmvalue.expost.P, state.investment)
    copyto!(welfare.V, state.welfare)
    copyto!(welfare.P, state.taxes)

    @. firmvalue.expost.P = clamp(firmvalue.expost.P, φlims[1], φlims[2])
    @. welfare.P = clamp(welfare.P, τlims[1], τlims[2])
    @. firmvalue.continuation.P = T(NaN)

    return firmvalue, welfare
end

"Apply one joint Bellman update for the firm given the current government policy, and vice versa."
function equilibriumstep!(nextfirmvalue::FV, nextwelfare::TW, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal)
    governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, exantegrid, pricespace, firm, government, signal)

    return nextfirmvalue, nextwelfare
end

function equilibriumupdateerrors(nextfirmvalue::FV, nextwelfare::TW, firmvalue::FV, welfare::TW, valtol, poltol) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    εᶠᵛ = max(
        maximum(abs, nextfirmvalue.continuation.V .- firmvalue.continuation.V),
        maximum(abs, nextfirmvalue.expost.V .- firmvalue.expost.V),
    )
    εᶠₚ = maximum(abs, nextfirmvalue.expost.P .- firmvalue.expost.P)
    εʷᵛ = maximum(abs, nextwelfare.V .- welfare.V)
    εʷₚ = maximum(abs, nextwelfare.P .- welfare.P)
    ε = normalizederror(max(εᶠᵛ, εʷᵛ), max(εᶠₚ, εʷₚ), valtol, poltol)

    return εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε
end

function relaxequilibriumupdate!(firmvalue::FV, welfare::TW, nextfirmvalue::FV, nextwelfare::TW, τᶜ, relax) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    @. firmvalue.continuation.V = relax * nextfirmvalue.continuation.V + (1 - relax) * firmvalue.continuation.V
    @. firmvalue.expost.V = relax * nextfirmvalue.expost.V + (1 - relax) * firmvalue.expost.V
    @. firmvalue.expost.P = relax * nextfirmvalue.expost.P + (1 - relax) * firmvalue.expost.P
    @. welfare.V = relax * nextwelfare.V + (1 - relax) * welfare.V
    @. welfare.P = relax * nextwelfare.P + (1 - relax) * welfare.P

    @. firmvalue.expost.P = clamp(firmvalue.expost.P, 0, 1)
    @. welfare.P = clamp(welfare.P, 0, 2τᶜ)
    @. firmvalue.continuation.P = T(NaN)

    return firmvalue, welfare
end

function dampedequilibriumwarmstart!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; maxiter = 0, relax = 0.1, minrelax = 1e-4, maxrelax = 0.5, shrink = 0.5, grow = 1.25, valtol = 1e-8, poltol = 1e-4, verbose = 0) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    maxiter == 0 && return 0, firmvalue, welfare

    nextfirmvalue = similar(firmvalue)
    nextwelfare = similar(welfare)
    candidatefirmvalue = similar(firmvalue)
    candidatewelfare = similar(welfare)
    trialfirmvalue = similar(firmvalue)
    trialwelfare = similar(welfare)
    relax = T(relax)
    minrelax = T(minrelax)
    maxrelax = T(maxrelax)
    shrink = T(shrink)
    grow = T(grow)

    for iter in 1:maxiter
        equilibriumstep!(nextfirmvalue, nextwelfare, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal)
        εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumupdateerrors(nextfirmvalue, nextwelfare, firmvalue, welfare, valtol, poltol)

        if verbose > 1
            @printf "Warm-start iteration %d: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e, normalized = %.2e\n" iter εᶠᵛ εᶠₚ εʷᵛ εʷₚ ε
        end

        if ε < 1
            relaxequilibriumupdate!(firmvalue, welfare, nextfirmvalue, nextwelfare, τᶜ, one(T))
            return iter, firmvalue, welfare
        end

        accepted = false
        λ = relax

        while λ >= minrelax
            copyto!(candidatefirmvalue, firmvalue)
            copyto!(candidatewelfare, welfare)
            relaxequilibriumupdate!(candidatefirmvalue, candidatewelfare, nextfirmvalue, nextwelfare, τᶜ, λ)

            equilibriumstep!(trialfirmvalue, trialwelfare, candidatefirmvalue, candidatewelfare, τᶜ, exantegrid, pricespace, firm, government, signal)
            _, _, _, _, εcandidate = equilibriumupdateerrors(trialfirmvalue, trialwelfare, candidatefirmvalue, candidatewelfare, valtol, poltol)

            if εcandidate < ε
                copyto!(firmvalue, candidatefirmvalue)
                copyto!(welfare, candidatewelfare)
                relax = min(maxrelax, grow * λ)
                accepted = true
                break
            end

            λ *= shrink
        end

        if !accepted
            if verbose > 0
                @warn @sprintf "Warm-start stopped at iteration %d because no damped update reduced the residual\n" iter
            end

            return iter - 1, firmvalue, welfare
        end
    end

    return maxiter, firmvalue, welfare
end

function equilibriumresidualerrors(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal, valtol, poltol) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    nextfirmvalue = similar(firmvalue)
    nextwelfare = similar(welfare)
    equilibriumstep!(nextfirmvalue, nextwelfare, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal)

    return equilibriumupdateerrors(nextfirmvalue, nextwelfare, firmvalue, welfare, valtol, poltol)
end

"Split the structured residual into firm and government value and policy blocks and return the associated errors."
function equilibriumerrors(residual::EquilibriumState, valtol, poltol)
    εΨᵛ = maximum(abs, residual.continuation)
    εᵛ = maximum(abs, residual.expost)
    εᶠₚ = maximum(abs, residual.investment)
    εʷᵛ = maximum(abs, residual.welfare)
    εʷₚ = maximum(abs, residual.taxes)
    εᶠᵛ = max(εΨᵛ, εᵛ)
    ε = normalizederror(max(εᶠᵛ, εʷᵛ), max(εᶠₚ, εʷₚ), valtol, poltol)

    return εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε
end

struct EquilibriumResidualScales{T}
    continuation::T
    expost::T
    investment::T
    welfare::T
    taxes::T
end

function residualblockscale(A, floor)
    return max(floor, maximum(abs, A))
end

function EquilibriumResidualScales(x::EquilibriumState, τᶜ)
    T = eltype(x)
    valuefloor = one(T)
    policyfloor = one(T)
    taxfloor = max(one(T), abs(2τᶜ))

    return EquilibriumResidualScales(
        residualblockscale(x.continuation, valuefloor),
        residualblockscale(x.expost, valuefloor),
        residualblockscale(x.investment, policyfloor),
        residualblockscale(x.welfare, valuefloor),
        residualblockscale(x.taxes, taxfloor),
    )
end

EquilibriumResidualScales(::Nothing, ::Any) = nothing

function scaleequilibriumresidual!(residual::EquilibriumState, scales::EquilibriumResidualScales)
    @. residual.continuation = residual.continuation / scales.continuation
    @. residual.expost = residual.expost / scales.expost
    @. residual.investment = residual.investment / scales.investment
    @. residual.welfare = residual.welfare / scales.welfare
    @. residual.taxes = residual.taxes / scales.taxes

    return residual
end

scaleequilibriumresidual!(residual::EquilibriumState, ::Nothing) = residual

mutable struct EquilibriumResidualParameters{Tτ, G, TP, TF, TG, TS, TV, TPOL, TRS}
    τᶜ::Tτ
    exantegrid::G
    pricespace::TP
    firm::TF
    government::TG
    signal::TS
    valtol::TV
    poltol::TPOL
    scales::TRS
    traceblocks::Bool
    traceevery::Int
    evaluations::Int
end

function EquilibriumResidualParameters(
    τᶜ,
    exantegrid,
    pricespace,
    firm,
    government,
    signal;
    valtol,
    poltol,
    scales = nothing,
    traceblocks = false,
    traceevery = 1,
)
    return EquilibriumResidualParameters(
        τᶜ,
        exantegrid,
        pricespace,
        firm,
        government,
        signal,
        valtol,
        poltol,
        scales,
        traceblocks,
        max(traceevery, 1),
        0,
    )
end

function traceequilibriumresidual!(parameters::EquilibriumResidualParameters, residual::EquilibriumState)
    parameters.evaluations += 1

    if parameters.traceblocks && parameters.evaluations % parameters.traceevery == 0
        εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumerrors(residual, parameters.valtol, parameters.poltol)
        @printf "Residual eval %d: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e, normalized = %.2e\n" parameters.evaluations εᶠᵛ εᶠₚ εʷᵛ εʷₚ ε
    end

    return nothing
end

"Evaluate the fixed-point residual `T(x) - x` of the synchronous firm-government Bellman update."
function rawequilibriumresidual!(residual::EquilibriumState, x::EquilibriumState, parameters::EquilibriumResidualParameters)
    firmvalue = firmvaluefromstate(x)
    welfare = welfarefromstate(x, parameters.τᶜ)
    nextfirmvalue = firmvaluefromstate(residual)
    nextwelfare = welfarefromstate(residual, parameters.τᶜ)

    equilibriumstep!(nextfirmvalue, nextwelfare, firmvalue, welfare, parameters.τᶜ, parameters.exantegrid, parameters.pricespace, parameters.firm, parameters.government, parameters.signal)

    @. residual.continuation = residual.continuation - x.continuation
    @. residual.expost = residual.expost - x.expost
    @. residual.investment = residual.investment - x.investment
    @. residual.welfare = residual.welfare - x.welfare
    @. residual.taxes = residual.taxes - x.taxes

    return residual
end

"Evaluate the scaled fixed-point residual used by `NonlinearSolve.jl`."
function equilibriumresidual!(residual::EquilibriumState, x::EquilibriumState, parameters::EquilibriumResidualParameters)
    rawequilibriumresidual!(residual, x, parameters)
    traceequilibriumresidual!(parameters, residual)
    scaleequilibriumresidual!(residual, parameters.scales)

    return residual
end

"Solve the structured equilibrium residual system with `NonlinearSolve.jl`."
function nonlinearequilibrium!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; algorithm::SciMLBase.AbstractNonlinearAlgorithm = LimitedMemoryBroyden(threshold = 6, max_resets = 25, linesearch = NonlinearSolve.LiFukushimaLineSearch()), maxiter = 100, valtol = 1e-8, poltol = 1e-4, φlims = (zero(T), one(T)), τlims = (zero(T), 2τᶜ), verbose = 0, traceblocks = false, traceevery = 1, scaleresiduals = true, acceptworse = false, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}

    x₀ = EquilibriumState(firmvalue, welfare)
    scales = scaleresiduals ? EquilibriumResidualScales(x₀, τᶜ) : nothing
    parameters = EquilibriumResidualParameters(
        τᶜ,
        exantegrid,
        pricespace,
        firm,
        government,
        signal;
        valtol,
        poltol,
        scales,
        traceblocks,
        traceevery,
    )

    initialresidual = similar(x₀)
    rawequilibriumresidual!(initialresidual, x₀, parameters)
    _, _, _, _, initialε = equilibriumerrors(initialresidual, valtol, poltol)

    nonlinearproblem = NonlinearProblem{true}(equilibriumresidual!, x₀, parameters)

    sol = solve(nonlinearproblem, algorithm; abstol = min(valtol, poltol), reltol = zero(T), maxiters = maxiter, show_trace = Val(verbose > 1), kwargs...)

    residual = similar(sol.u)
    rawequilibriumresidual!(residual, sol.u, parameters)
    εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumerrors(residual, valtol, poltol)

    if !acceptworse && ε > initialε
        if verbose > 0
            @warn @sprintf "Rejected NonlinearSolve update because normalized residual increased from %.2e to %.2e\n" initialε ε
        end

        return sol, firmvalue, welfare
    end

    unpackequilibrium!(firmvalue, welfare, sol.u; φlims, τlims)

    if verbose > 0
        @printf "NonlinearSolve finished with residual errors: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e\n" εᶠᵛ εᶠₚ εʷᵛ εʷₚ
    end

    if ε ≥ 1 && verbose > 0
        @warn @sprintf "Equilibrium residual still above tolerance after NonlinearSolve: normalized error = %.2e\n" ε
    end

    return sol, firmvalue, welfare
end

"Call `nonlinearequilibrium!` with `NonlinearSolve.LimitedMemoryBroyden`."
function broydenequilibrium!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; memory = 6, max_resets = 25, linesearch = NonlinearSolve.LiFukushimaLineSearch(), kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    algorithm = LimitedMemoryBroyden(threshold = memory, max_resets = max_resets, linesearch = linesearch)

    return nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; algorithm, kwargs...)
end

"Run `nonlinearequilibrium!` along a path of signal-noise levels using the previous stage as the initial condition."
function homotopynonlinear!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; σpath::TP = [signal.σ], algorithm::SciMLBase.AbstractNonlinearAlgorithm = LimitedMemoryBroyden(threshold = 6, max_resets = 25, linesearch = NonlinearSolve.LiFukushimaLineSearch()), warmstartiters = 0, warmstartrelax = 0.1, broydenstarttol = Inf, maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}, TP <: AbstractVector{T}}
    initialsignal = Signal(signal.μ, σpath[1], signal.space)
    setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, initialsignal)
    setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, initialsignal)

    solutions = []
    for (i, σᵢ) in enumerate(σpath)
        if verbose > 0
            @printf "\nHomotopy %d/%d, σ = %.4f\n" i length(σpath) σᵢ
        end

        signalᵢ = Signal(signal.μ, σᵢ, signal.space)
        dampedequilibriumwarmstart!(
            firmvalue,
            welfare,
            τᶜ,
            exantegrid,
            pricespace,
            firm,
            government,
            signalᵢ;
            maxiter = warmstartiters,
            relax = warmstartrelax,
            valtol,
            poltol,
            verbose,
        )

        εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumresidualerrors(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signalᵢ, valtol, poltol)
        if verbose > 0
            @printf "Post-warm-start residual: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e, normalized = %.2e\n" εᶠᵛ εᶠₚ εʷᵛ εʷₚ ε
        end

        if ε > broydenstarttol
            if verbose > 0
                @warn @sprintf "Skipping NonlinearSolve at σ = %.4f because warm-start residual %.2e exceeds broydenstarttol %.2e\n" σᵢ ε broydenstarttol
            end

            push!(solutions, nothing)
            continue
        end

        sol, _, _ = nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signalᵢ; algorithm, maxiter, valtol, poltol, verbose, kwargs...)
        push!(solutions, sol)
    end

    return solutions, firmvalue, welfare
end
