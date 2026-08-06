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
import SciMLBase, DiffEqBase
import ForwardDiff, DiffResults

# Optimization
import Optimization, OptimizationOptimJL
import ADTypes, DifferentiationInterface
import Roots

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
firm, government, signal, climate = initmodels()

Δm = 150firm.e₀ # 150 years without abatement
m̄ = climate.m₀ + Δm

lb = [climate.m₀, firm.a₀]
ub = [m̄, firm.e₀]

## Solve
## Nonlinear constrained optimization problem setup
function committedobjective(x, p)
    firm, government, climate = p
    mₛ, aₛ = x

    return J(mₛ, aₛ, firm, government, climate)
end

committedparameters = (firm, government, climate);
x₀ = @. (ub - lb) / 2
committedobjectivefunction = Optimization.OptimizationFunction(committedobjective, ADTypes.AutoForwardDiff())

committedproblem = Optimization.OptimizationProblem(committedobjectivefunction, x₀, committedparameters; lb, ub)

committedsolution = Optimization.solve(committedproblem, OptimizationOptimJL.BFGS())