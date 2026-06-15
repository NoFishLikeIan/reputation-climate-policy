## Setup
using Revise
using UnPack
using Optim
using Printf
using JLD2
using LinearAlgebra

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/concentrationboundary.jl")

## Defaults
firm = StaticFirm()
government = Government()
signal = Signal()
climate = Climate()

τᶜ = computeτᶜ(climate, government, firm)

mgrid = range(0, 2.5, 101)

## Solve boundary problems
rightsolution = rightboundary(mgrid, τᶜ, climate, government, firm)
leftsolution = solveleftboundary(mgrid, τᶜ, climate, government, firm; verbose = true)

@printf "Committed tax τᶜ = %.4e\n" τᶜ
@printf "Removal rate δₘ = %.4e\n" climate.δₘ
@printf "Right boundary residual = %.4e\n" rightsolution.residual
@printf "Left boundary residual = %.4e\n" leftsolution.residual
@printf "Left boundary converged = %s in %d iterations\n" string(leftsolution.converged) leftsolution.iterations

## Save
mkpath(joinpath("data", "solutions"))
solutionpath = joinpath("data", "solutions", "concentrationboundary.jld2")

JLD2.@save solutionpath leftsolution rightsolution mgrid τᶜ climate signal government firm
