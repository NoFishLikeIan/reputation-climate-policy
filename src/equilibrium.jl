using NonlinearSolve
using Printf

"Stores the ranges used to pack the joint equilibrium objects into one optimisation vector."
struct EquilibriumLayout{R <: AbstractRange{Int}}
    Ψ::R
    v::R
    φ::R
    w::R
    τ::R
end

"Construct the packing layout associated with the current dimensions of `firmvalue` and `welfare`."
function EquilibriumLayout(firmvalue::FV, welfare::TW) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    start = 1

    nΨ = length(firmvalue.continuation.V)
    Ψ = start:start + nΨ - 1
    start = last(Ψ) + 1

    nv = length(firmvalue.expost.V)
    v = start:start + nv - 1
    start = last(v) + 1

    nφ = length(firmvalue.expost.P)
    φ = start:start + nφ - 1
    start = last(φ) + 1

    nw = length(welfare.V)
    w = start:start + nw - 1
    start = last(w) + 1

    nτ = length(welfare.P)
    τ = start:start + nτ - 1

    return EquilibriumLayout(Ψ, v, φ, w, τ)
end

Base.length(layout::EquilibriumLayout) = last(layout.τ)

"Pack `firmvalue` and `welfare` into the vector `x` using the ranges stored in `layout`."
function packequilibrium!(x::AbstractVector, firmvalue::FV, welfare::TW, layout::EquilibriumLayout) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    copyto!(view(x, layout.Ψ), vec(firmvalue.continuation.V))
    copyto!(view(x, layout.v), vec(firmvalue.expost.V))
    copyto!(view(x, layout.φ), vec(firmvalue.expost.P))
    copyto!(view(x, layout.w), vec(welfare.V))
    copyto!(view(x, layout.τ), vec(welfare.P))

    return x
end

"Unpack the optimisation vector `x` into `firmvalue` and `welfare`, and project the policy blocks onto the admissible sets."
function unpackequilibrium!(firmvalue::FV, welfare::TW, x::AbstractVector, layout::EquilibriumLayout; φlims = (0., Inf), τlims = (0., Inf)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}}
    copyto!(vec(firmvalue.continuation.V), view(x, layout.Ψ))
    copyto!(vec(firmvalue.expost.V), view(x, layout.v))
    copyto!(vec(firmvalue.expost.P), view(x, layout.φ))
    copyto!(vec(welfare.V), view(x, layout.w))
    copyto!(vec(welfare.P), view(x, layout.τ))

    @. firmvalue.expost.P = clamp(firmvalue.expost.P, φlims[1], φlims[2])
    @. welfare.P = clamp(welfare.P, τlims[1], τlims[2])
    @. firmvalue.continuation.P = T(NaN)

    return firmvalue, welfare
end

"Apply one joint Bellman update for the firm given the current government policy, and vice versa."
function equilibriumstep!(nextfirmvalue::FV, nextwelfare::TW, firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; φlims = (0., 1.), τlims = (0., 2τᶜ)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    firmstep!(nextfirmvalue, firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, signal; φlims)
    governmentstep!(nextwelfare, welfare, firmvalue, τᶜ, exantegrid, pricespace, firm, government, signal; τlims)

    return nextfirmvalue, nextwelfare
end

"Split the packed residual into firm and government value and policy blocks and return the associated errors."
function equilibriumerrors(residual::AbstractVector, layout::EquilibriumLayout, valtol, poltol)
    εΨᵛ = maximum(abs, view(residual, layout.Ψ))
    εᵛ = maximum(abs, view(residual, layout.v))
    εᶠₚ = maximum(abs, view(residual, layout.φ))
    εʷᵛ = maximum(abs, view(residual, layout.w))
    εʷₚ = maximum(abs, view(residual, layout.τ))
    εᶠᵛ = max(εΨᵛ, εᵛ)
    ε = normalizederror(max(εᶠᵛ, εʷᵛ), max(εᶠₚ, εʷₚ), valtol, poltol)

    return εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε
end

"Stores the parameters and workspaces needed to evaluate the packed equilibrium residual in place."
struct EquilibriumResidual{L, FV, TW, TV, G, TP, TF, TG, TS, TLφ, TLτ}
    layout::L
    firmvalue::FV
    welfare::TW
    nextfirmvalue::FV
    nextwelfare::TW
    nextx::TV
    τᶜ
    exantegrid::G
    pricespace::TP
    firm::TF
    government::TG
    signal::TS
    φlims::TLφ
    τlims::TLτ
end

"Construct the parameter object and work arrays used by `NonlinearSolve.jl`."
function EquilibriumResidual(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; φlims = (0., 1.), τlims = (0., 2τᶜ)) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    layout = EquilibriumLayout(firmvalue, welfare)
    nextx = Vector{T}(undef, length(layout))
    φlimst = (T(φlims[1]), T(φlims[2]))
    τlimst = (T(τlims[1]), T(τlims[2]))

    return EquilibriumResidual(
        layout,
        similar(firmvalue),
        similar(welfare),
        similar(firmvalue),
        similar(welfare),
        nextx,
        τᶜ,
        exantegrid,
        pricespace,
        firm,
        government,
        signal,
        φlimst,
        τlimst,
    )
end

"Evaluate the fixed-point residual `T(x) - x` of the synchronous firm-government Bellman update."
function equilibriumresidual!(residual::AbstractVector, x::AbstractVector, problem::EquilibriumResidual)
    unpackequilibrium!(problem.firmvalue, problem.welfare, x, problem.layout; φlims = problem.φlims, τlims = problem.τlims)
    equilibriumstep!(
        problem.nextfirmvalue,
        problem.nextwelfare,
        problem.firmvalue,
        problem.welfare,
        problem.τᶜ,
        problem.exantegrid,
        problem.pricespace,
        problem.firm,
        problem.government,
        problem.signal;
        φlims = problem.φlims,
        τlims = problem.τlims,
    )
    packequilibrium!(problem.nextx, problem.nextfirmvalue, problem.nextwelfare, problem.layout)
    @. residual = problem.nextx - x

    return residual
end

"Solve the packed equilibrium residual system with `NonlinearSolve.jl`."
function nonlinearequilibrium!(
    firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; 
    algorithm::AbstractNonlinearAlgorithm = LimitedMemoryBroyden(threshold = 10), maxiter = 100, valtol = 1e-8, poltol = 1e-4, φlims = (zero(T), one(T)), τlims = (zero(T), 2τᶜ), verbose = 0, kwargs...
) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}

    problem = EquilibriumResidual(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; φlims, τlims)
    x0 = Vector{T}(undef, length(problem.layout))
    packequilibrium!(x0, firmvalue, welfare, problem.layout)

    nonlinearproblem = NonlinearProblem{true}(equilibriumresidual!, x0, problem)

    sol = solve(nonlinearproblem, algorithm; abstol = min(valtol, poltol), reltol = zero(T), maxiters = maxiter, show_trace = Val(verbose > 1), kwargs...)

    unpackequilibrium!(firmvalue, welfare, sol.u, problem.layout; φlims, τlims)

    residual = similar(sol.u)
    equilibriumresidual!(residual, sol.u, problem)
    εᶠᵛ, εᶠₚ, εʷᵛ, εʷₚ, ε = equilibriumerrors(residual, problem.layout, valtol, poltol)

    if verbose > 0
        @printf "NonlinearSolve finished with residual errors: firm value = %.2e, firm policy = %.2e, welfare value = %.2e, welfare policy = %.2e\n" εᶠᵛ εᶠₚ εʷᵛ εʷₚ
    end

    if ε >= one(ε) && verbose > 0
        @warn @sprintf "Equilibrium residual still above tolerance after NonlinearSolve: normalized error = %.2e\n" ε
    end

    return sol, firmvalue, welfare
end

"Call `nonlinearequilibrium!` with `NonlinearSolve.LimitedMemoryBroyden`."
function broydenequilibrium!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; memory = 10, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}}
    algorithm = LimitedMemoryBroyden(threshold = memory)

    return nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; algorithm, memory, kwargs...)
end

"Run `nonlinearequilibrium!` along a path of signal-noise levels using the previous stage as the initial condition."
function homotopynonlinear!(firmvalue::FV, welfare::TW, τᶜ, exantegrid::G, pricespace, firm::Firm, government::Government, signal::Signal; σpath::TP = [signal.σ], algorithm = nothing, memory = 10, verbose = 0, kwargs...) where {T, FV <: FirmValue{T}, TW <: ValueFunction{2, T}, G <: Grid{2}, TP <: AbstractVector{T}}
    initialsignal = Signal(signal.μ, σpath[1], signal.space)
    setfirmboundaries!(firmvalue, τᶜ, exantegrid, pricespace, firm, initialsignal)
    setgovernmentboundaries!(welfare, τᶜ, exantegrid, firm, government, initialsignal)

    solutions = Vector{Any}(undef, length(σpath))
    for (i, σᵢ) in enumerate(σpath)
        if verbose > 0
            @printf "\nHomotopy %d/%d, σ = %.4f\n" i length(σpath) σᵢ
        end

        signalᵢ = Signal(signal.μ, σᵢ, signal.space)
        algorithmᵢ = isnothing(algorithm) ? LimitedMemoryBroyden(threshold = memory) : algorithm
        sol, _, _ = nonlinearequilibrium!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signalᵢ; algorithm = algorithmᵢ, memoryverbose, kwargs...)
        solutions[i] = sol
    end

    return solutions, firmvalue, welfare
end