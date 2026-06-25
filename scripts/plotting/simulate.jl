## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots

import FastInterpolations as Itp
import JLD2
import Random
import Statistics
import UnPack: @unpack, @pack!
import FastClosures: @closure

import DifferentialEquations as DE
import StochasticDiffEq as SDE

import Base.Threads

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/primitives/climate.jl")
includet("../../src/agents/government.jl")
includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("../../src/solve/equilibrium.jl")
includet("../../src/dynamics/state.jl")
includet("../../src/utils/analysis.jl")

includet("utils.jl")
publicationdefaults!()

const SIMPATH = joinpath("data", "solutions")

## Load data
firm, government, signal, climate = initmodels()

filename = solutionlabel(climate, government, firm, signal)
solutionpath = joinpath(SIMPATH, "$filename.jld2")
figurepath = joinpath("figures", filename)
mkpath(figurepath)

solutionfile = JLD2.jldopen(solutionpath)
@unpack committedpolicy, mgrid = solutionfile["committed"]
committedmgrid = mgrid
@unpack φgrid, mgrid, interiorpolicy = solutionfile["interior"]
close(solutionfile)

## Interpolation
τᶜ = Itp.linear_interp(committedmgrid, committedpolicy; extrap = Itp.ClampExtrap())
τ = Itp.linear_interp((φgrid, mgrid), interiorpolicy; extrap = Itp.ClampExtrap())

## Simulate
parameters = (τ, τᶜ, government, firm, signal)
φ₀grid = [0.1, 0.4, 0.5, 0.6, 0.9]
horizon = 100.
timegrid = 0:horizon
trajectories = 1_000

Random.seed!(11148705)

solutions = map(φ₀grid) do φ₀
    prob = SDE.SDEProblem(F!, G!, [φ₀, m₀], (0, horizon), parameters)
    ensembleprob = SDE.EnsembleProblem(prob)
    solution = SDE.solve(ensembleprob; trajectories)
end;

function computepolicies(x, p, t)
    τ, τᶜ, government, firm, _ = p
    φ, m = x

    τᶜₜ = τᶜ(m)
    τₜ = τ(x)
    eₜ = e(aᵇ(τₜ, φ, τᶜₜ, government, firm), firm)

    return (τ = τₜ, τᶜ = τᶜₜ, e = eₜ)
end

timesteps = range(0, horizon; step = 1 / 12)
simulatedpolicies = map(solution -> computeoverensemble(solution, computepolicies, timesteps), solutions);

## Figures
φ₀palette = Plots.palette(:viridis, length(φ₀grid));
percentageformatter = @closure x -> @sprintf "%.0f%%" 100x
φlabel(φ) = latexstring("\\phi_0 = " * @sprintf("%.2f", φ))

## Tax policy
begin
    τfig = Plots.plot(
        xlabel = "Year",
        ylabel = "Carbon tax [USD / tCO2e]",
        legend_title = L"\phi_0",
        xlims = extrema(timesteps),
    )
    
    for (i, φ₀) in enumerate(φ₀grid)

        policies = simulatedpolicies[i]
        τmedian = vec(Statistics.median(getindex.(policies, :τ), dims = 1)) ./ taxfactor

        Plots.plot!(timesteps, τmedian; label = φlabel(φ₀), c = φ₀palette[i])
    end

    safesavefigure(τfig, joinpath(figurepath, "simulation-tax-median.png"))

    τfig
end
