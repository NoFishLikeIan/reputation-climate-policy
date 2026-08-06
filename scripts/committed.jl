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
import ADTypes
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
lowerbound = SA.SVector(climate.m₀, firm.a₀, firm.a₀)
upperbound = SA.SVector(m̄, firm.e₀, firm.e₀)

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
    m, a, ā = x

    res[1] = committedfeasibility(a, m, firm, government, climate)
    res[2] = (ā - a) / (firm.e₀ - firm.a₀)
end

function computefeasiblepoint(m, margin, firm, government, climate)
    Roots.find_zero(
        a -> committedfeasibility(a, m, firm, government, climate) - margin,
        (firm.a₀, firm.e₀)
    )
end

## Solve problem
committedparameters = (firm, government, climate);
feasibilitymargin = 5e-2
x₀ = [ climate.m₀, computefeasiblepoint(climate.m₀, feasibilitymargin, firm, government, climate), firm.e₀ ]
adtype = ADTypes.AutoForwardDiff()

committedobjectivefunction = Optimization.OptimizationFunction(committedobjective, adtype; cons = constraints)

committedproblem = Optimization.OptimizationProblem(
    committedobjectivefunction,
    x₀,
    committedparameters;
    lb = lowerbound,
    ub = upperbound,
    lcons = [1e-4, 0.],
    ucons = [Inf, Inf]
)

committedsolution = Optimization.solve(
    committedproblem,
    OptimizationIpopt.IpoptOptimizer();
    hessian_approximation = "limited-memory",
    check_derivatives_for_naninf = "yes",
    bound_relax_factor = 0.
)
