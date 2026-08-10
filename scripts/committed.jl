## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings
using Plots

import JLD2

import Base.Threads
import FastClosures: @closure
import UnPack: @unpack, @pack!

# Linear algebra
import LinearAlgebra as LA
import SparseArrays
import StaticArrays as SA
import StaticArraysCore

# Interpolation and integration
import SciMLBase
import SpecialFunctions
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqRosenbrock as ODERosenbrock
import BoundaryValueDiffEq as BVP

# Optimization
import Optimization, OptimizationOptimJL
import ADTypes, DifferentiationInterface
import Optim, OptimizationNLopt

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/dynamics/state.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/firm/committed.jl")
includet("../src/solve/government/committed.jl")

includet("plotting/utils.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm = Firm()
government = Government()
climate = Climate()

parameters = CommittedParameters(firm, government, climate)
scaling = ScalingParameters(parameters)

## Solve
y0 = [1., 100., 0.9 * firm.e₀]
lb = [0., 0., 0.5 * firm.e₀]
ub = [150., 150., firm.e₀]

optparameters = (parameters, scaling)
committedobjective(y0, optparameters)
timingconstraints(y0, optparameters)

## Solve
adtype = DifferentiationInterface.SecondOrder(ADTypes.AutoFiniteDiff(), ADTypes.AutoFiniteDiff())
committedobjectivefunction = Optimization.OptimizationFunction(committedobjective, adtype)
netzeroproblem = Optimization.OptimizationProblem(committedobjectivefunction, y0, optparameters; lb = lb, ub = ub)

partialsolution = Optimization.solve(netzeroproblem, OptimizationOptimJL.LBFGS(); maxiters = 10_000)

## Plot net-zero solution
starttimegrid = range(0., 50; step = 1)
endtimegrid = range(30, 100; step = 1.)

yopt = partialsolution.u

let
    objfigure = contourf(starttimegrid, endtimegrid, (tₛ, t̄) -> committedobjective(CommittedState(tₛ, t̄, yopt[3]), optparameters); xlabel = L"t_s", ylabel = L"\bar{t}", linewidth = 0, c = :viridis)
    scatter!(objfigure, yopt[[1]], yopt[[2]]; c = :black, label = false)
end