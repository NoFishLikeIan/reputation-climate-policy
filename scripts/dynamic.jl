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
import DomainSets
import MethodOfLines as MOL
import ModelingToolkit as MTK

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
firm = DynamicFirm(ν = ν₀)
government = Government()
signal = Signal()

## Committed policy
terminaltime = 100.
tsteps = 1200
tgrid = range(0., terminaltime, tsteps)
τᶜtraj = [computeτᶜ(t, government, firm) for t in tgrid]

## Static terminal guess
τᶜterminal = last(τᶜtraj)
terminalfirm = StaticFirm(terminaltime, firm)

terminalsolutions = solvestaticmassmatrix(τᶜterminal, signal, government, terminalfirm; verbose = true, εs = [1e-2, 1e-3])

_, terminalφgrid, terminalvalue = terminalsolutions[end]
φgrid = vcat(0., terminalφgrid, 1.)
φspan = extrema(φgrid)
φnodes = length(φgrid)

## Dynamic HJB PDE
uₜ = vcat(
    leftcost(terminaltime, τᶜterminal, government, firm),
    first.(terminalvalue),
    rightcost(terminaltime, τᶜterminal, government, firm),
)

setdynamicpdecontext!(terminaltime, φgrid, uₜ, tgrid, τᶜtraj, signal, government, firm)

MTK.@parameters s φ
MTK.@variables U(..)
MTK.@variables u u′ u′′

MTK.@register_symbolic dynamicpdeflow(s, φ, u, u′, u′′)
MTK.@register_symbolic dynamicpdeterminal(φ)
MTK.@register_symbolic dynamicpdeleftvalue(s)
MTK.@register_symbolic dynamicpderightvalue(s)

∂ₛ  = MTK.Differential(s)
∂φ = MTK.Differential(φ)
∂²φ = ∂φ^2

equations = [
    ∂ₛ(U(s, φ)) ~ dynamicpdeflow(s, φ, U(s, φ), ∂φ(U(s, φ)), ∂²φ(U(s, φ))),
]

conditions = [
    U(0., φ) ~ dynamicpdeterminal(φ),
    U(s, 0.) ~ dynamicpdeleftvalue(s),
    U(s, 1.) ~ dynamicpderightvalue(s),
]

domains = [
    s ∈ DomainSets.Interval(0., terminaltime),
    φ ∈ DomainSets.Interval(0., 1.),
]

MTK.@named dynamicpdesystem = MTK.PDESystem(equations, conditions, domains, [s, φ], [U(s, φ)])
discretisation = MOL.MOLFiniteDifference([φ => collect(φgrid)], s; approx_order = 2)
pdeproblem = MOL.discretize(dynamicpdesystem, discretisation)

@printf "Solving dynamic HJB PDE on %d time nodes and %d belief nodes\n" tsteps φnodes

pdesolution = ODE.solve(pdeproblem, BDF.QNDF(autodiff = false); saveat = tgrid, abstol = 1e-6, reltol = 1e-6, progress = true)
sgrid = pdesolution[s]
φgrid = pdesolution[φ]
solutiongrid = pdesolution[U(s, φ)]

if !SciMLBase.successful_retcode(pdesolution.original_sol)
    @warn "Dynamic HJB PDE failed with retcode $(pdesolution.original_sol.retcode)"
end

solutionpath = joinpath("data", "solutions", dynamicsolutionlabel(firm))

JLD2.@save solutionpath solutiongrid sgrid φgrid tgrid τᶜtraj signal government firm terminalsolutions
