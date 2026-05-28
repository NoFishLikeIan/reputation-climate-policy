## Setup
using Revise
using BenchmarkTools
using Printf

import UnPack: @unpack
import FastClosures: @closure

import Optim
import JLD2

import Roots
import FastInterpolations

import SciMLBase
import OrdinaryDiffEqBDF as BDF
import BoundaryValueDiffEq as BVP
import OrdinaryDiffEq as ODE

includet("../src/utils/saving.jl")
includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/valuefunction.jl")
includet("../src/solve/staticproblem.jl")
includet("../src/solve/dynamicvaluefunction.jl")

## Defaults
firm = DynamicFirm(ω = 5e-2, ν = 4e-2)
government = Government()
signal = Signal(σ = 0.20666 / 10)

## Committed policy
terminaltime = 100.
tsteps = 1200
tgrid = range(0., terminaltime, tsteps)
τᶜtraj = [computeτᶜ(t, government, firm) for t in tgrid]

## Terminal condition
τᶜterminal = last(τᶜtraj)
terminalfirm = StaticFirm(terminaltime, firm)
terminalνsteps = defaultνsteps(terminalfirm)

terminalνcontinuation = solvestaticνcontinuation(τᶜterminal, signal, government, terminalfirm; verbose = true, φsteps = [1e-3], νsteps = terminalνsteps)

terminalsolutions = terminalνcontinuation[end].solutions

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
prob = ODE.ODEProblem(dynamicHJB!, uₜ, tspan, parameters)
solution = ODE.solve(prob, ODE.Tsit5(); saveat = reverse(tgrid), abstol = 1e-6, reltol = 1e-6, progress = true)

if !SciMLBase.successful_retcode(solution.retcode)
    @warn "Dynamic HJB failed with retcode $(solution.retcode)"
end

## Save
solution = SciMLBase.strip_solution(solution)
solutionpath = joinpath("data", "solutions", dynamicsolutionlabel(firm))

JLD2.@save solutionpath solution ℓgrid tgrid τᶜtraj signal government firm terminalνcontinuation
