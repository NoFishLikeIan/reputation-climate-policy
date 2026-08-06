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
import Optimization, OptimizationIpopt
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
lowerbound = SA.SVector(firm.a₀, climate.m₀)
upperbound = SA.SVector(firm.e₀, climate.m₀ + Δm)

## Illustrate optimization problem at ā
agrid = range(firm.a₀, firm.e₀, 201)
mgrid = climate.m₀ .+ range(0, Δm, 200)

contourf(agrid, mgrid, (a, m) -> begin
        if singularity∂ₐM(a, m, firm, government, climate) ≤ 1e-6
            return NaN
        else
            return J(m, a, firm.e₀, firm, government, climate)
        end
    end;
    linewidth = 0.    
)


## Solve
## Nonlinear constrained optimization problem setup
function committedobjective(x, p)
    firm, government, climate = p
    mₛ, aₛ, ā = x

    return J(mₛ, aₛ, ā, firm, government, climate)
end

function constraints(res, x, p)
    firm, government, climate = p
    m, a = x[1:2]

    res[1] = singularity∂ₐM(a, m, firm, government, climate)
end

function computefeasiblepoint(m, firm, government, climate)
    Roots.find_zero(a -> singularity∂ₐM(a, m, firm, government, climate) - 1e-8, (firm.a₀, firm.e₀))
end

## Solve problem
committedparameters = (firm, government, climate);
x₀ = [climate.m₀, computefeasiblepoint(climate.m₀, firm, government, climate), firm.e₀]
adtype = DifferentiationInterface.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())

committedobjectivefunction = Optimization.OptimizationFunction(committedobjective, adtype; cons = constraints)

committedproblem = Optimization.OptimizationProblem(committedobjectivefunction, x₀, (firm, government, climate); lcons = [1e-8], ucons = [Inf])

committedsolution = Optimization.solve(committedproblem, OptimizationIpopt.IpoptOptimizer())