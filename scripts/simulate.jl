## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = "#FAFAFA", size = 400 .* (√2, 1))

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
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/simulate/state.jl")

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

## Drift of beliefs
begin
    mlevels = (1., 1.5, 2.)
    φgrid = range(0, 1, 1001)

    dφfig = Plots.plot(xlims = (0, 1))

    for factor in mlevels
        m = factor * m₀
        Plots.plot!(φgrid, φ -> dφ(φ, τ(φ, m), τᶜ(m), signal); label = factor)
    end

    dφfig
end

## Simulate
parameters = τ, τᶜ, firm, signal
x₀ = [0.01, m₀]
horizon = 1000.

prob = SDE.SDEProblem(F!, G!, x₀, (0., horizon), parameters)

begin
    simfig = Plots.plot()
    for φ₀ in [0.02, 0.1, 0.2, 0.4, 0.6, 0.99]
        sol = SDE.solve(prob; u0 = [φ₀, m₀])
        Plots.plot!(sol, idxs = 2, label = φ₀)
    end
    simfig
end
