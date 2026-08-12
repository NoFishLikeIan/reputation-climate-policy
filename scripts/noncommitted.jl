## Setup
using Revise
using Printf

using LaTeXStrings
using Plots

import FastInterpolations as Itp
import JLD2
import UnPack: @unpack

# Linear algebra
import SparseArrays
import StaticArrays as SA

# Interpolation and integration
import ADTypes
import SciMLBase
import SpecialFunctions
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqBDF as BDF
import BoundaryValueDiffEq as BVP

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/dynamics/state.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/government/committed.jl")
includet("../src/solve/government/noncommitted.jl")

includet("plotting/utils.jl")

## Load committed problem
firm = Firm()
government = Government()
climate = Climate()

signal = Signal()

committedlabel = solutionlabel(climate, government, firm)
committedfile = joinpath("data", "solutions", "singular", "$committedlabel.jld2")
committedsolution = JLD2.load(committedfile)

trajectory = committedsolution["trajectory"]
committedtaxes = committedsolution["taxes"]
committedtime = committedsolution["time"]

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)

activecommittedtax = Itp.linear_interp(committedtime, committedtaxes; extrap = Itp.ClampExtrap())
τᶜ = CommittedTaxPath(activecommittedtax, activeterminal, terminal, terminalabatement, firm, government)

## State space
φgrid = range(0., 1., 21)
agrid = range(firm.a₀, firm.e₀, 21)

# The padding prevents the upper m boundary from entering the domain of
# dependence of the initial state.
mpadding = 1.25 * e(firm.a₀, firm) * terminal
mgrid = range(climate.m₀, climate.m₀ + mpadding, 31)

grid = NonCommittedGrid(φgrid, mgrid, agrid)
parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate)

## Solve backwards from the end of the committed tax tail
@printf "Solving %d firm equations and %d government equations over %.1f years\n" length(grid) length(grid) terminal

taxswitch = (terminal - activeterminal) / terminal
tstops = iszero(taxswitch) ? Float64[] : [taxswitch]

solution = solvenoncommitted(
    parameters;
    saveat = range(0., 1.; length = 101),
    tstops,
    abstol = 1e-6,
    reltol = 1e-6,
)

if !SciMLBase.successful_retcode(solution)
    error("Non-committed solution failed with retcode $(solution.retcode)")
end

initialstate = last(solution.u)
initialpolicies = noncommittedpolicies(initialstate, parameters, 1.)
initialvalues = noncommittedvalues(initialstate, parameters)
initialkkt = noncommittedkktdiagnostics(initialstate, parameters, 1.)

@printf(
    "Initial KKT violations: firm %.3e, tax %.3e, complementarity %.3e\n",
    initialkkt.firmviolation,
    initialkkt.taxviolation,
    initialkkt.complementarity,
)

## Plot initial policies
if isinteractive()
    mindex = firstindex(mgrid)
    aindex = firstindex(agrid)

    taxfigure = plot(
        φgrid,
        initialpolicies.tax[:, mindex, aindex] ./ taxfactor;
        xlabel = L"Reputation $\phi$",
        ylabel = "USD / tCO2e",
        label = L"Non-committed tax $\tau$",
        linewidth = 2.,
    )

    investmentfigure = plot(
        φgrid,
        initialpolicies.investment[:, mindex, aindex];
        xlabel = L"Reputation $\phi$",
        ylabel = "GtCO2e / year²",
        label = L"Investment $u$",
        linewidth = 2.,
    )

    plot(taxfigure, investmentfigure; layout = (1, 2), size = (900, 360))
end
