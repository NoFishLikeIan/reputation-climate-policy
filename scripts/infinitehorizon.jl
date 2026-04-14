## Modules
using Revise

using FastInterpolations
using FastGaussQuadrature
using Optim

## Imports
includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/signal.jl")

includet("../src/grid.jl")
includet("../src/valuefunction.jl")
includet("../src/pfi.jl")

## Setup
ns = (100, 101, 102)
abatementspace = range(0, 10, ns[1])
logitspace, _ = gausshermite(ns[2])
stategrid = Grid((abatementspace, logitspace))

signalspace = range(0, 1.5, ns[3])
statesignalgrid = Grid((abatementspace, logitspace, signalspace))

firm = Firm()
government = Government()

## Value Function
welfare = ValueFunction(stategrid, (1., 0.))
firmvalue = ValueFunction(statesignalgrid, (1., 0.))

## Iteration