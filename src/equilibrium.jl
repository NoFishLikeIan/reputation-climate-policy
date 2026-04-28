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

    @. firmvalue.expost.P = clamp(firmvalue.expost.P, 0, 1)
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

"Evaluate the fixed-point residual `T(x) - x` of the synchronous firm-government Bellman update."
function equilibriumresidual!(residual::EquilibriumState, x::EquilibriumState, parameters)
    τᶜ, exantegrid, pricespace, firm, government, signal = parameters

    firmvalue = firmvaluefromstate(x)
    welfare = welfarefromstate(x, τᶜ)
    nextfirmvalue = firmvaluefromstate(residual)
    nextwelfare = welfarefromstate(residual, τᶜ)

    equilibriumstep!(nextfirmvalue, nextwelfare, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal)

    @. residual = residual - x

    return residual
end

"Solve the packed equilibrium residual system with `NonlinearSolve.jl`."
function nonlinearequilibrium!(
    firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; 
    algorithm::SciMLBase.AbstractNonlinearAlgorithm = LimitedMemoryBroyden(threshold = 10), maxiter = 100, valtol = 1e-8, poltol = 1e-4, verbose = 0, kwargs...
) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}

    parameters = (τᶜ, exantegrid, pricespace, firm, government, signal)
    x₀ = EquilibriumState(firmvalue, welfare)

    nonlinearproblem = NonlinearProblem{true}(equilibriumresidual!, x₀, parameters)

    sol = solve(nonlinearproblem, algorithm; abstol = min(valtol, poltol), reltol = zero(T), maxiters = maxiter, show_trace = Val(verbose > 1), kwargs...)

    unpackequilibrium!(firmvalue, welfare, sol.u)

    residual = similar(sol.u)
    equilibriumresidual!(residual, sol.u, parameters)
    εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumerrors(residual, valtol, poltol)

    if verbose > 0
        @printf "NonlinearSolve finished with residual errors: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e\n" εᶠᵛ εᶠₚ εʷᵛ εʷₚ
    end

    if ε ≥ 1 && verbose > 0
        @warn @sprintf "Equilibrium residual still above tolerance after NonlinearSolve: normalized error = %.2e\n" ε
    end

    return sol, firmvalue, welfare
end

"Call `nonlinearequilibrium!` with `NonlinearSolve.LimitedMemoryBroyden`."
function broydenequilibrium!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; memory = 10, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    algorithm = LimitedMemoryBroyden(threshold = memory)

    return nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; algorithm, kwargs...)
end

"Run `nonlinearequilibrium!` along a path of signal-noise levels using the previous stage as the initial condition."
function homotopynonlinear!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; σpath::TP = [signal.σ], algorithm::SciMLBase.AbstractNonlinearAlgorithm = LimitedMemoryBroyden(), verbose = 0, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}, TP <: AbstractVector{T}}
    initialsignal = Signal(signal.μ, σpath[1], signal.space)
    setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, initialsignal)
    setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, initialsignal)

    solutions = []
    for (i, σᵢ) in enumerate(σpath)
        if verbose > 0
            @printf "\nHomotopy %d/%d, σ = %.4f\n" i length(σpath) σᵢ
        end

        signalᵢ = Signal(signal.μ, σᵢ, signal.space)
        sol, _, _ = nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signalᵢ; algorithm, verbose, kwargs...)
        push!(solutions, sol)
    end

    return solutions, firmvalue, welfare
end
