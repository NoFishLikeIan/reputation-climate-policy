## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
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
import FastChebInterp
import FastGaussQuadrature
import OrdinaryDiffEq as ODE
import SciMLBase, DiffEqBase

# Optimization and root finding
import Optimization
import OptimizationOptimJL, OptimizationNLopt
import ForwardDiff

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/dynamics/state.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/root.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/utils.jl")
includet("../src/solve/firm/committed.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm, government, signal, climate = initmodels()

## Chebyshev collocation grid
orders = [(n, n) for n in 1:6]
Δm = 150firm.e₀ # 150 years without abatement
m̄ = climate.m₀ + Δm
lowerbound = SA.SVector(firm.a₀, climate.m₀)
upperbound = SA.SVector(firm.e₀, climate.m₀ + Δm)

## Approximate the committed tax
const taxscale = firm.r * c(firm.e₀, firm)

τᶜinitguess = @closure u -> (u[2] / upperbound[2]) * defaultscc

function governmentobjective(η, optparameters)
    firm, government, climate, lb, ub = optparameters
    n = isqrt(length(η))
    coefficients = taxscale .* reshape(η, n, n)
    τᶜ = FastChebInterp.ChebPoly(coefficients, lb, ub)

    return welfarecosts(τᶜ, firm, government, climate)
end

optparameters = (firm, government, climate, lowerbound, upperbound)
objectivefunction = SciMLBase.OptimizationFunction(governmentobjective)

points = FastChebInterp.chebpoints(first(orders), lowerbound, upperbound)
initialvalues = map(τᶜinitguess, points)
initialpolicy = FastChebInterp.chebinterp(initialvalues, lowerbound, upperbound; tol = 0)

continuation = solvechebyshevcontinuation(objectivefunction, initialpolicy, orders, lowerbound, upperbound, optparameters; coefficientscale = taxscale, coefficientlower = -0.5, coefficientupper = 0.5)

continuation.converged || @warn "Chebyshev continuation stopped before the final order"
sol = continuation.solution

## Plot optimal policy
import Plots

τᶜopt = continuation.policy
agrid = range(firm.a₀, firm.e₀, 1001)
mgrid = range(climate.m₀, m̄, 1001)

Plots.contourf(agrid, mgrid, (a, m) -> τᶜopt(SA.SVector(a, m)))
