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

optparameters = (parameters, scaling)

## Solve
y0 = [80., firm.e₀ * 0.9]
lb = [10., firm.a₀]
ub = [100., firm.e₀]

objectivefunction = @closure (y, ∇) -> begin
    if length(∇) > 0
        FiniteDiff.finite_difference_gradient!(
            ∇, y -> committedobjective(y, optparameters), y
        )
    end

    return committedobjective(y, optparameters)
end

opt = NLopt.Opt(:LN_COBYLA, length(y0))
NLopt.lower_bounds!(opt, lb)
NLopt.upper_bounds!(opt, ub)
NLopt.xtol_rel!(opt, 1e-8)
NLopt.min_objective!(opt, objectivefunction)

objective, yopt, ret = NLopt.optimize(opt, y0)
yopt = CommittedState(yopt...)

## Plot optimisation problem
durationgrid = range(50., 100.; step = 0.5)
abatementgrid = range(firm.a₀, firm.e₀; step = 0.5)

if isinteractive()
    objfigure = contourf(durationgrid, abatementgrid, (t̄, ā) -> committedobjective([t̄, ā], optparameters); xlabel = L"\bar{t}", ylabel = L"\bar{a}", linewidth = 0, c = :viridis)
    scatter!(objfigure, [yopt.t̄], [yopt.ā]; c = :white, label = false)
end

## Plot solution path
trajectory, taxes, time, terminalhamiltonian = committedpathdiagnostics(yopt, parameters, scaling);
@printf "Terminal hamiltonian %.5e" terminalhamiltonian

abatement = getindex.(trajectory, 2)

if isinteractive()
    afig = plot(time, abatement ./ firm.e₀; ylims = (0, 1), xlabel = "Year", label = "Fraction of abated emissions", xlims = extrema(time), c = :darkgreen, legend = :topleft, linewidth = 2.)

    taxfig = twinx(afig)

    plot!(taxfig, time, taxes ./ taxfactor, xlims = extrema(time), c = :darkred, label = L"Tax $\tau$", ylabel = "USD / tCO2e", legend = :bottomright, linewidth = 2.)
    hline!(taxfig, [0.], linestyle = :dot, label = false, c = :darkred)

    afig
end

## Save 
savelabel = solutionlabel(climate, government, firm)
savepath = "data/solutions/singular/"

filename = joinpath(savepath, "$savelabel.jld2")

JLD2.jldopen(filename, "w") do file
    @pack! file = trajectory, taxes, time, climate, government, firm
end