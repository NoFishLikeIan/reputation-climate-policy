using Revise

import Printf
import JLD2
import UnPack: @unpack 
import OrdinaryDiffEq as ODE
import SciMLBase
import FastInterpolations as Itp

import LinearSolve
import SparseArrays
import StaticArrays as SA
import StochasticDiffEq as SDE
import OrdinaryDiffEq as ODE

using Plots, LaTeXStrings
Plots.default(dpi = 180, label = false, linewidth = 2.)

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

function dynamicdrift(x, dynamicparameters, t)
    solution, parameters, grid = dynamicparameters
    φ = clamp(x[1], 0, 1)
    m = x[2]
    a = x[3]

    τₜ, τᶜₜ, uₜ = policy(t, (φ, m, a), solution, parameters, grid)

    dφ = beliefdrift(χ(τₜ, τᶜₜ, parameters.signal), φ)
    dm = cumulativeemissionsdrift(a, parameters.firm)
    da = uₜ

    return SA.SVector(dφ, dm, da)
end

function dynamicnoise(x, dynamicparameters, t)
    solution, parameters, grid = dynamicparameters
    φ = clamp(x[1], 0, 1)
    m = x[2]
    a = x[3]

    τₜ, τᶜₜ, _ = policy(t, (φ, m, a), solution, parameters, grid)
    σᵩ = beliefdiffusion(χ(τₜ, τᶜₜ, parameters.signal), φ)
    
    return SA.SVector(σᵩ, 0, 0)
end

x₀ = SA.SVector(0.5, climate.m₀, firm.a₀)
dynamicparameters = (solution, parameters, grid);

dynamicfn = SDE.SDEFunction{false}(dynamicdrift, dynamicnoise)
dynamicprob = SDE.SDEProblem(dynamicfn, x₀, (0, parameters.horizon), dynamicparameters)
φs = (0.1, 0.5, 0.75, 0.9, 1.0)
solutions = [SDE.solve(dynamicprob, SDE.SRIW1(); u0 = SA.SVector(φ₀, climate.m₀, firm.a₀)) for φ₀ in φs];

begin
    abatementfig = hline(ylims = (0, firm.e₀), xlabel = L"t", ylabel = L"a")
    belieffigure = hline(ylims = (0, 1), xlabel = L"t", ylabel = L"\phi")

    for (i, φ₀) in enumerate(φs)
        dynamicsol = solutions[i]
        plot!(abatementfig, dynamicsol; idxs = 3, label = φ₀)
        plot!(belieffigure, dynamicsol; idxs = 1, label = φ₀)
    end

    plot(belieffigure, abatementfig; margins = 5Plots.mm, size = 500 .* (2√2, 1))
end