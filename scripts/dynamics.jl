using Revise

import JLD2
import UnPack: @unpack 
import OrdinaryDiffEq as ODE
import SciMLBase
import FastInterpolations as Itp

using Plots
Plots.default(dpi = 180, label = false, linewidth = 2.)

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/government/committed.jl")
includet("../src/solve/government/noncommitted.jl")

## Load problem
## Save 
filename = "e04.190e01_kappa8.453e-02_xi1.718e00_firmdiscount7.000e-02_y01.972e02_r2.000e-02_gamma1.000e-02_zeta4.800e-04_epsilon1.000e00_sigma3.800e-01.jld2"
solpath = joinpath("data", "solutions", "uncommitted", filename)
if !isfile(solpath) throw("File $solpath not found.") end

file = JLD2.jldopen(solpath, "r")
@unpack solution, grid, firm, government, climate, signal, taxmethod = file
close(file)

committedlabel = solutionlabel(climate, government, firm)
committedfile = joinpath("data", "solutions", "committed", "$committedlabel.jld2")
committedsolution = JLD2.load(committedfile)
trajectory = committedsolution["trajectory"]
committedtaxes = committedsolution["taxes"]
committedtime = committedsolution["time"]

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)

activecommittedtax = Itp.linear_interp(committedtime, committedtaxes; extrap = Itp.ClampExtrap())
τᶜ = CommittedTaxPath(activecommittedtax, activeterminal, terminal, terminalabatement, firm, government)

parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate, taxmethod)

## Simulate path
function policy(t, x, solution, parameters::NonCommittedParameters, grid::NonCommittedGrid)
    φ, m, a = x
    s = noncommittedreversetime(t, parameters)

    policystate = solution(s)
    policies = noncommittedpolicies(policystate, parameters, s)
    
    τₜ = Itp.linear_interp((grid.φgrid, grid.mgrid, grid.agrid), policies.tax, (φ, m, a))
    τᶜₜ = parameters.τᶜ(t)
    uₜ = Itp.linear_interp((grid.φgrid, grid.mgrid, grid.agrid), policies.investment, (φ, m, a))    

    return (τₜ, τᶜₜ, uₜ)
end

φ₀ = rand()
x₀ = (φ₀, climate.m₀, firm.a₀)

