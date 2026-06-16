## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = :transparent, size = 400 .* (√2, 1))

import FastInterpolations as Itp
import JLD2
import Random
import UnPack: @unpack, @pack!

import StochasticDiffEq as SDE

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/simulate/state.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm = Firm()
government = Government()
signal = Signal()
climate = Climate()

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
parameters = τ, τᶜ, firm, signal
x₀ = [0.99, m₀]
horizon = 1000.

prob = SDE.SDEProblem(F!, G!, x₀, (0., horizon), parameters)
sol = SDE.solve(prob)
