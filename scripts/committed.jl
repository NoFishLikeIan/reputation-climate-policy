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
import NLopt
import FiniteDiff

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
y0 = [10., 80., firm.e₀ * 0.9]
lb = [0., 0., 0.]
ub = [150., 150., firm.e₀]

optparameters = (parameters, scaling)
committedobjective(y0, optparameters)
FiniteDiff.finite_difference_gradient(y -> committedobjective(y, optparameters), y0)

## Solve
objectivefunction = @closure (y, ∇) -> begin
    if length(∇) > 0
        FiniteDiff.finite_difference_gradient!(∇, y -> committedobjective(y, optparameters), y)
    end

    return committedobjective(y, optparameters)
end

objectiveconstraints = @closure (y, ∇) -> begin
    if length(∇) > 0
        ∇[1] = 1.
        ∇[2] = -1.
    end

    return y[1] - y[2] # tₛ - t̄ ≤ 0 
end

begin # Define optimisation problem
    opt = NLopt.Opt(:LN_COBYLA, 3)
    NLopt.lower_bounds!(opt, lb)
    NLopt.upper_bounds!(opt, ub)
    NLopt.xtol_rel!(opt, 1e-8)
    NLopt.min_objective!(opt, objectivefunction)
    NLopt.inequality_constraint!(opt, objectiveconstraints)
end

objective, yopt, ret = NLopt.optimize(opt, y0);
yopt = CommittedState(yopt...)

## Plot optimisation problem
starttimegrid = range(0., 50.; step = 0.5)
endtimegrid = range(50., 100.; step = 0.5)

let
    objfigure = contourf(starttimegrid, endtimegrid, (tₛ, t̄) -> committedobjective(CommittedState(tₛ, t̄, yopt[3]), optparameters); xlabel = L"t_s", ylabel = L"\bar{t}", linewidth = 0, c = :viridis)
    scatter!(objfigure, yopt[[1]], yopt[[2]]; c = :black, label = false)
end

## Plot solution path
pathparameters = CommittedPathParameters(yopt, parameters, scaling)
x0 = committedinitialguess(0., pathparameters)

problem = BVP.TwoPointBVProblem{true}(
    committednormaliseddrift!,
    (initialcondition!, terminalcondition!),
    x0,
    (0., 1.),
    pathparameters;
    bcresid_prototype = (zeros(SA.MVector{3}), zeros(SA.MVector{4}))
)

solutionpath = BVP.solve(problem, BVP.MIRK4(); dt = 1e-2)
trajectory = [physicalstate(u, scaling) for u in solutionpath.u]
