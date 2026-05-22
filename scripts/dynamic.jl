## Setup
using Revise
using Optim
using UnPack
using FastClosures
using BenchmarkTools
using Printf
using JLD2

using Roots
using FastInterpolations
using DifferentialEquations, OrdinaryDiffEqSDIRK, BoundaryValueDiffEq
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
terminaltime = 100.
tsteps = 1200
tgrid = range(0., terminaltime; length = tsteps)
τᶜtraj = [computeτᶜ(t, government, firm) for t in tgrid]

## Terminal condition
τᶜterminal = last(τᶜtraj)
terminalfirm = StaticFirm(terminaltime, firm)
terminalsolutions = solvestaticproblem(τᶜtraj[end], signal, government, terminalfirm; verbose = true)

_, ℓgrid, terminalvalue = terminalsolutions[end]
uₜ = first.(terminalvalue)
ℓspan = extrema(ℓgrid)
φspan = belief.(ℓspan)

## Test the PDE right hand side
parameters = (ℓgrid, tgrid, τᶜtraj, signal, government, firm)
duₜ = similar(uₜ)
dynamicHJB!(duₜ, uₜ, parameters, terminaltime)

## Solve backwards in time
φnodes = length(ℓgrid)
@printf "Solving dynamic HJB on %d time nodes and %d belief nodes\n" tsteps φnodes

tspan = (terminaltime, 0.)
prob = ODEProblem(dynamicHJB!, uₜ, tspan, parameters)
solution = solve(prob, TRBDF2(); saveat = reverse(tgrid), abstol = 1e-6, reltol = 1e-6)

if !successful_retcode(solution.retcode)
    @warn "Dynamic HJB failed with retcode $(solution.retcode)"
end

## Save
JLD2.@save "data/solutions/dynamic.jld2" solution ℓgrid tgrid τᶜtraj signal government firm
