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
import ADTypes
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
y0 = [1., 30., firm.e₀ * 0.9]
lb = [0., 1., 0.]
ub = [149., 150., firm.e₀]

optparameters = (parameters, scaling)
committedobjective(y0, optparameters)

committedobjectivefunction = Optimization.OptimizationFunction(committedobjective, ADTypes.AutoFiniteDiff())
netzeroproblem = Optimization.OptimizationProblem(committedobjectivefunction, y0, optparameters; lb = lb, ub = ub)

partialsolution = Optimization.solve(netzeroproblem, OptimizationNLopt.NLopt.LD_LBFGS(); maxiters = 100)