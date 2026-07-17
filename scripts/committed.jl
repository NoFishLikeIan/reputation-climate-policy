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
import Optimization, OptimizationOptimJL
import ADTypes, ForwardDiff

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
order = (1, 1)
Δm = 150firm.e₀ # 150 years without abatement
m̄ = climate.m₀ + Δm
lowerbound = SA.SVector(firm.a₀, climate.m₀)
upperbound = SA.SVector(firm.e₀, climate.m₀ + Δm)
grid = CommittedGrid(order, lowerbound, upperbound)

## Approximate the committed tax
const taxscale = firm.r * c(firm.e₀, firm)

τᶜinitguess = @closure u -> (u[2] / upperbound[2]) * defaultscc
τᶜ = FastChebInterp.chebinterp(τᶜinitguess.(grid.points), lowerbound, upperbound; tol = 0)

function governmentobjective(η, optparameters)
    firm, government, climate, lb, ub = optparameters

    τᶜ = FastChebInterp.ChebPoly(taxscale .* η , lb, ub)

    return welfarecosts(τᶜ, firm, government, climate)
end

η₀ = τᶜ.coefs ./ taxscale
optparameters = (firm, government, climate, lowerbound, upperbound)
governmentobjective(η₀, optparameters)

ηlower = fill(-1., size(η₀))
ηupper = fill(1., size(η₀))
fn = SciMLBase.OptimizationFunction(governmentobjective, ADTypes.AutoForwardDiff())
prob = SciMLBase.OptimizationProblem(fn, η₀, optparameters; lb = ηlower, ub = ηupper)

sol = Optimization.solve(prob, OptimizationOptimJL.LBFGS())