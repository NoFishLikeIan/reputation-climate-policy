## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = :transparent, size = 400 .* (√2, 1))

import FastInterpolations as Itp
import JLD2
import Random
import UnPack: @unpack, @pack!
import FastClosures: @closure

import StochasticDiffEq as SDE

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/dynamics/state.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm, government, signal, climate = initmodels()

filename = solutionlabel(climate, government, firm, signal)
solutionpath = joinpath(SIMPATH, "$filename.jld2")

solutionfile = JLD2.jldopen(solutionpath)
@unpack committedpolicy, mgrid = solutionfile["committed"]
committedmgrid = mgrid
@unpack φgrid, mgrid, interiorpolicy = solutionfile["interior"]
close(solutionfile)

## Interpolation
τᶜ = Itp.linear_interp(committedmgrid, committedpolicy; extrap = Itp.ClampExtrap())
τ = Itp.linear_interp((φgrid, mgrid), interiorpolicy; extrap = Itp.ClampExtrap())

## Simulate
parameters = τ, τᶜ, government, firm, signal
x₀ = [0.01, m₀]
horizon = 1000.

prob = SDE.SDEProblem(F!, G!, x₀, (0., horizon), parameters)

begin
    simfig = Plots.plot()
    
    for φ₀ in [0.01, 0.3, 0.5, 0.8, 0.99]
        sol = SDE.solve(prob; u0 = [φ₀, m₀])
        Plots.plot!(sol, idxs = 1, label = φ₀)
    end

    simfig
end
