## Setup
using Revise
using Optim
using UnPack
using FastClosures
using BenchmarkTools
using Printf
using JLD2

using DifferentialEquations, BoundaryValueDiffEq
using SciMLBase: successful_retcode

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/valuefunction.jl")
includet("../src/solve/staticproblem.jl")
includet("../src/solve/dynamicvaluefunction.jl")

## Defaults
firm = DynamicFirm()
government = Government()
signal = Signal()

## Committed policy
terminaltime = 300.
tsteps = 101
tgrid = range(0., terminaltime; length = tsteps)
τᶜtraj = [computeτᶜ(t, government, firm) for t in tgrid]

## Belief grid
φstep = 1e-4
φnodes = 401
φspan = (φstep, 1 - φstep)
ℓspan = logit.(φspan)
ℓgrid = collect(range(first(ℓspan), last(ℓspan); length = φnodes))

## Terminal condition
τᶜterminal = last(τᶜtraj)
terminalfirm = StaticFirm(terminaltime, firm)
uₜ = solvestaticproblem(τᶜtraj[end], signal, government, terminalfirm)

## Test the PDE right hand side
parameters = (ℓgrid, tgrid, τᶜtraj, signal, government, firm, sunkabatement)
duₜ = similar(uₜ)
dynamicHJB!(duₜ, uₜ, parameters, terminaltime)

## Solve backwards in time
@printf "Solving dynamic HJB on %d time nodes and %d belief nodes\n" tsteps φnodes

tspan = (terminaltime, 0.)
prob = ODEProblem(dynamicHJB!, uₜ, tspan, parameters)
sol = solve(
    prob,
    TRBDF2();
    saveat = reverse(tgrid),
    abstol = 1e-6,
    reltol = 1e-6,
)

if !successful_retcode(sol.retcode)
    @warn "Dynamic HJB failed with retcode $(sol.retcode)"
end

## Save
JLD2.@save "data/solutions/dynamic-continuous-time.jld2" sol ℓgrid tgrid τᶜtraj signal government firm sunkabatement
