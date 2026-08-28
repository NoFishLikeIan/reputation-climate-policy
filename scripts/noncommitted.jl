## Setup
using Revise
using Printf

using LaTeXStrings
using Plots

import FastInterpolations as Itp
import JLD2
import UnPack: @unpack

# Linear algebra
import LinearSolve
import SparseArrays
import StaticArrays as SA

# Interpolation and integration
import ADTypes
import SciMLBase, SciMLLogging, DiffEqBase
import SpecialFunctions
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqBDF as BDF
import BoundaryValueDiffEq as BVP

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/dynamics/state.jl")
includet("../src/dynamics/belief.jl")
includet("../src/dynamics/firm.jl")
includet("../src/dynamics/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/government/committed.jl")
includet("../src/solve/government/noncommitted.jl")

## Load committed problem
firm, government, signal, climate = initmodels()

committedfile = joinpath("data", "solutions", solutionfilename(climate, government, firm))
if !isfile(committedfile)
    error("Committed solution file $committedfile not found. Run scripts/committed.jl first.")
end

trajectory, committedtaxes, committedtime = JLD2.jldopen(committedfile, "r") do file
    file["trajectory"], file["taxes"], file["time"]
end

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)

activecommittedtax = Itp.linear_interp(committedtime, committedtaxes; extrap = Itp.ClampExtrap())
τᶜ = CommittedTaxPath(activecommittedtax, activeterminal, terminal, terminalabatement, firm, government)

taxmethod = OneShotTax()
## State space
ns = (101, 50, 49)

φgrid = range(0., 1., ns[1])

mpadding = 1.05 * e(firm.a₀, firm) * terminal # The padding prevents the upper m boundary from entering the domain of dependence of the initial state.
mgrid = range(climate.m₀, climate.m₀ + mpadding, ns[2])

agrid = range(firm.a₀, firm.e₀, ns[3])

grid = NonCommittedGrid(φgrid, mgrid, agrid)
parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate, taxmethod)

## Solve backwards from the end of the committed tax tail
@printf "Solving %d equations over %.1f years\n" length(grid) terminal

taxswitch = (terminal - activeterminal) / terminal
tstops = taxswitch > 0 ?  [taxswitch] : typeof(taxswitch)[]

problem = noncommittedproblem(parameters)
algorithm = BDF.FBDF(linsolve = LinearSolve.KrylovJL_GMRES(), concrete_jac = false)
verbosity = DiffEqBase.DEVerbosity(SciMLLogging.None())
solution = ODE.solve(problem, algorithm; abstol = 1e-6, reltol = 1e-6, verbose = verbosity)

if !SciMLBase.successful_retcode(solution)
    error("Non-committed solution failed with retcode $(solution.retcode)")
end

## Diagnostics
initialstate = last(solution.u)
initialpolicies = noncommittedpolicies(initialstate, parameters, 1.)
initialvalues = noncommittedvalues(initialstate, parameters)
initialkkt = noncommittedkktdiagnostics(initialstate, parameters, 1.)

initialcommittedtax = τᶜ(0.)
initialtaxrange = extrema(initialpolicies.tax) ./ taxfactor
initialexpectedtaxrange = extrema(initialpolicies.expectedtax) ./ taxfactor

@printf("Initial KKT violations: firm %.3e, tax %.3e, complementarity %.3e\n",
    initialkkt.firmviolation,
    initialkkt.taxviolation,
    initialkkt.complementarity)
@printf("Initial tax Hamiltonian gap: %.3e\n",
    initialkkt.taxhamiltoniangap)
@printf("At t = 0: committed tax %.3f, non-committed tax [%.3f, %.3f], expected current tax [%.3f, %.3f] USD / tCO2e\n",
    initialcommittedtax / taxfactor,
    initialtaxrange...,
    initialexpectedtaxrange...)

committedinvestment = getindex.(trajectory, 3)
committedindex = argmax(committedtaxes .* committedinvestment)
policytime = committedtime[committedindex]
policys = noncommittedreversetime(policytime, parameters)
policystate = solution(policys)
policies = noncommittedpolicies(policystate, parameters, policys)
policykkt = noncommittedkktdiagnostics(policystate, parameters, policys)
committedtaxatpolicy = τᶜ(policytime)

mindex = argmin(abs.(mgrid .- trajectory[committedindex][1]))
aindex = argmin(abs.(agrid .- trajectory[committedindex][2]))

expectedtaxerror = maximum(abs.(policies.expectedtax[end, :, :] .- committedtaxatpolicy))
policytaxrange = extrema(policies.tax) ./ taxfactor
policyexpectedtaxrange = extrema(policies.expectedtax) ./ taxfactor
policycoefficientrange = extrema(policies.taxcoefficient)
policycurvaturerange = extrema(policies.taxcurvature)

@printf(
    "At t = %.2f: committed tax %.3f, non-committed tax [%.3f, %.3f], expected current tax [%.3f, %.3f] USD / tCO2e\n",
    policytime,
    committedtaxatpolicy / taxfactor,
    policytaxrange...,
    policyexpectedtaxrange...,
)

@printf(
    "Tax coefficient [%.3e, %.3e]; tax curvature [%.3e, %.3e]\n",
    policycoefficientrange...,
    policycurvaturerange...,
)

@printf(
    "Tax KKT violation %.3e; Hamiltonian gap %.3e\n",
    policykkt.taxviolation,
    policykkt.taxhamiltoniangap,
)

@printf(
    "Maximum error in τᵉ(φ = 1) = τᶜ is %.3e\n",
    expectedtaxerror,
)

## Save 
solutionkey = uncommittedsolutionkey(signal, taxmethod)

JLD2.jldopen(committedfile, "a+") do file
    file["$solutionkey/solution"] = solution
    file["$solutionkey/grid"] = grid
    file["$solutionkey/signal"] = signal
    file["$solutionkey/taxmethod"] = taxmethod
end
