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
import NLopt
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

householdnl = Household(household.ν, 1 + eps(Float64), household.r)
modelnl = ModelParameters(householdnl, firm, government, climate)

filename = joinpath(SIMPATH, solutionfilename(household, firm, government, climate))

if isfile(filename)
    throw("Committed solution in $filename already saved! Breaking to avoid overwriting.")
end

## Define and hot-start ODE problem
x₀ = initialguess(SA.MVector, model)

p = CommittedPathParameters(100., model)
odeprob = ODE.ODEProblem(driftcommitted!, x₀, (0., p.T), p)
abatemnetcallback = ODE.DiscreteCallback(isfullabatement, ODE.terminate!)

odealg = ODE.AutoTsit5(ODE.Rosenbrock23())
odesol = ODE.solve(odeprob, odealg)

## Define and solve BVP problem
bvpalg = BVP.MIRK4()
lb, ub = optimisationbounds(SA.MVector, model; λmax = Inf)

bcresid_prototype = (initialcondition(x₀, p), terminalcondition(x₀, p))
problem = BVP.TwoPointBVProblem(driftcommitted!, (initialcondition!, terminalcondition!), odesol, (0., p.T), p; lb = lb, ub = ub, bcresid_prototype)

solution = BVP.solve(problem, bvpalg; dt = p.T * 1e-2)

initialcondition(solution.u[1], p)
terminalcondition(solution.u[end], p)

