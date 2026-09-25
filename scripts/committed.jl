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
import OrdinaryDiffEqCore as ODECore
import OrdinaryDiffEqRosenbrock as ODERosenbrock
import BoundaryValueDiffEq as BVP
import QuadGK
import SpecialFunctions as SF

# Optimization
import Optimization, Optim
import FiniteDiff
import Roots
import BandedMatrices: BandError

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/households.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/dynamics/state.jl")
includet("../src/dynamics/firm.jl")
includet("../src/dynamics/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")
includet("../src/utils/integrals.jl")

includet("../src/solve/government/committed.jl")

includet("plotting/utils.jl")

const SIMPATH = joinpath("data", "solutions")
ispath(SIMPATH) || mkpath(SIMPATH);

## Defaults
household, firm, government, signal, climate = initmodels()
model = ModelParameters(household, firm, government, climate)

filename = joinpath(SIMPATH, solutionfilename(household, firm, government, climate))

if isfile(filename)
    throw("Committed solution in $filename already saved! Breaking to avoid overwriting.")
end

## Define and hot-start ODE problem
x₀ = initialguess(SA.MVector, model)
odeproblem = ODE.ODEProblem(ODE.ODEFunction{true}(driftcommitted!), x₀, (0., 1.), model)

z₀ = SA.MVector{7}(x₀..., 0.)
welfareproblem = ODE.ODEProblem(ODE.ODEFunction{true}(driftwelfarecommitted!), z₀, (0., 1.), model)

## Solver
lb, ub = optimisationbounds(model)
bcresid_prototype = (initialcondition(x₀, model), terminalcondition(x₀, model))

bvpfunction = SciMLBase.BVPFunction(driftcommitted!, (initialcondition!, terminalcondition!); bcresid_prototype, twopoint = Val(true))
bvproblem = BVP.TwoPointBVProblem(bvpfunction, ODE.solve(odeproblem, defodealg), odeproblem.tspan, model; lb, ub)

## Solve
objectivewelfare(1., bvproblem, welfareproblem, odeproblem, model)
optimisationparameters = (bvproblem, welfareproblem, odeproblem, model, 0.05, defbvpalg, defodealg);

optsol = Optim.optimize(Base.Fix2(objectivewelfare, optimisationparameters), 0., 150., Optim.Brent())
T = optsol.minimizer
optimalpath = BVP.solve(bvproblem, defbvpalg; u0 = ODE.solve(odeproblem, defodealg; tspan = T), dt = 1e-2T, tspan = (0., T))

taxpath = map(Base.Fix2(committedtax, model), optimalpath.u)
ts = optimalpath.t

τᶜ = CommittedTaxPath(taxpath, ts)

## Illustrate solution


## Save solution
