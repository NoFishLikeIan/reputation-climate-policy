## Setup
using Revise
using Optim
using UnPack
using FastClosures
using BenchmarkTools
using Printf
using JLD2
import FastInterpolations

using DifferentialEquations, BoundaryValueDiffEq
import BoundaryValueDiffEq as BVP
import SciMLBase
using SciMLBase: successful_retcode

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/valuefunction.jl")
includet("../src/solve/staticproblem.jl")

## Defaults
firm = StaticFirm()
government = Government()
signal = Signal()

τᶜ = computeτᶜ(government, firm)
parameters = (τᶜ, signal, government, firm)

α = leftboundaryexponent(parameters)
@printf "Left boundary exponent α = %.4e\n" α

## Solve value function
εs = [1e-2, 1e-3, 1e-4]
solutions = solvestaticmassmatrix(τᶜ, signal, government, firm; εs, verbose = true)

JLD2.@save "data/solutions/static.jld2" solutions εs τᶜ signal government firm
