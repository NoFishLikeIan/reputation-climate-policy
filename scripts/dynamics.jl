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

import LaTeXStrings: @L_str
import Plots
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

includet("../src/dynamics/simulation.jl")

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
policies = constructpolicies(solution, parameters, grid)

## Simulate path
x₀ = SA.SVector(0.5, climate.m₀, firm.a₀)
dynamicparameters = (policies, parameters, grid);
endtime = activeterminal

dynamicfn = SDE.SDEFunction{false}(dynamicdrift, dynamicnoise)
dynamicprob = SDE.SDEProblem(dynamicfn, x₀, (0, endtime), dynamicparameters)
ensembleproblem = SDE.EnsembleProblem(dynamicprob)

φs = [0.1, 0.2, 0.5, 0.75, 0.9, 1.0]
EnsemblePolicy = Vector{Vector{NTuple{3, Float64}}}
solutions = SciMLBase.EnsembleSolution[]
policyensembles = EnsemblePolicy[] 
for φ₀ in φs
    Printf.@printf "Solving φ₀ = %.1f\r" φ₀
    sol = SDE.solve(ensembleproblem, SDE.SOSRI(); u0 = SA.SVector(φ₀, climate.m₀, firm.a₀), trajectories = 500)

    policyensemble = Vector{NTuple{3, Float64}}[]
    for soli in sol.u
        policytraj = [ policy(t, u, policies, parameters, grid) for (t, u) in zip(soli.t, soli.u) ]
        push!(policyensemble, policytraj)
    end

    push!(solutions, sol)
    push!(policyensembles, policyensemble)
end

## Plot
figurepath = joinpath("figures", solutionlabel(climate, government, firm, signal))
!ispath(figurepath) && mkpath(figurepath)
let
    nφ = length(φs)
    beliefcolors = Plots.palette(:Dark2_3, nφ)
    beliefsfigures = Plots.Plot[]
    concentrationfigures = Plots.Plot[]
    abatemnetfigures = Plots.Plot[]
    taxfigures = Plots.Plot[]

    for (i, φ₀) in enumerate(φs)
        Printf.@printf "Plotting φ₀ = %.1f\r" φ₀
        dynamicsol = solutions[i]
        c = beliefcolors[i]

        # State
        belieffigure = Plots.plot(ylims = (0, 1), xlabel = "Year", title = L"$\phi_0 = %$(φ₀)$")
        concentrationfig = Plots.plot(ylims = extrema(grid.mgrid), xlabel = "Year", ylabel = "GtCO2", title = L"$\phi_0 = %$(φ₀)$")
        abatementfigure = Plots.plot(ylims = (0, firm.e₀), xlabel = "Year", ylabel = "GtCO2 per year", title = L"$\phi_0 = %$(φ₀)$")

        Plots.plot!(belieffigure, dynamicsol; idxs = 1, alpha = 0.25, c)
        Plots.plot!(concentrationfig, dynamicsol; idxs = 2, alpha = 0.25, c)
        Plots.plot!(abatementfigure, dynamicsol; idxs = 3, alpha = 0.25, c)

        push!(beliefsfigures, belieffigure)
        push!(concentrationfigures, concentrationfig)
        push!(abatemnetfigures, abatementfigure)

        # Policy
        policyensemble = policyensembles[i]
        timetraj = dynamicsol.u[1].t
        τᶜtraj = getindex.(policyensemble[1], 2) ./ taxfactor
        taxfigure = Plots.plot(timetraj, τᶜtraj; c, linestyle = :dash, xlabel = "Year", ylabel = "USD per tCO2")

        for (i, policytraj) in enumerate(policyensemble)
            τtraj = getindex.(policytraj, 1)
            Plots.plot!(dynamicsol.u[i].t, τtraj ./ taxfactor; alpha = 0.25, c)
        end

        push!(taxfigures, taxfigure)
    end

    rows = isqrt(nφ)
    columns = nφ - rows

    beliefsfigjoint = Plots.plot(beliefsfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Belief $\phi$", ylims = (0, 1))
    concentrationfigjoint = Plots.plot(concentrationfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Concentration $m$", ylims = extrema(grid.mgrid))
    abatemnetfigjoint = Plots.plot(abatemnetfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Abatemnet $a$", ylims = (0, firm.e₀))
    taxfigjoint = Plots.plot(taxfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Tax $\tau$")

    Plots.savefig(beliefsfigjoint, joinpath(figurepath, "beliefs.png"))
    Plots.savefig(concentrationfigjoint, joinpath(figurepath, "concentration.png"))
    Plots.savefig(abatemnetfigjoint, joinpath(figurepath, "abatement.png"))
    Plots.savefig(taxfigjoint, joinpath(figurepath, "tax.png"))

    println("Saved figures in ", figurepath)
end