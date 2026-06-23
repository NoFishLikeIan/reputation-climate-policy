## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots

import FastClosures: @closure
import UnPack: @unpack, @pack!

import JLD2
import LinearAlgebra as LA
import SparseArrays as SA
import Optim

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/utils.jl")
includet("../src/solve/equilibrium.jl")
includet("../src/solve/committedvalue.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm, government, signal, climate = initmodels()

Δm = 500firm.e₀ # 500 years without abatement
mgrid = range(m₀, m₀ + Δm, 1001);

## Initialise value function problem
uᶜ₀ = [w(m, 0.01, a(0.01, government, firm), climate, government, firm) for m in mgrid]
uᶜ = copy(uᶜ₀)

_, (i, abserror, relerror) = solvehjb!(uᶜ, mgrid, climate, government, firm; maxiters = 100_000, verbose = 1, abstol = 1e-8, reltol = 1e-6, Δt⁻¹ = 10.)

committedpolicy = computeglobalpolicy(uᶜ, mgrid, government, firm)
filename = solutionlabel(climate, government, firm, signal)
solutionpath = joinpath(SIMPATH, "$filename.jld2")
figurepath = joinpath("figures", filename)
mkpath(figurepath)

JLD2.jldopen(solutionpath, "a+") do file
    if haskey(file, "committed") 
        delete!(file, "committed")
    end

    solution = JLD2.Group(file, "committed")
    @pack! solution = mgrid, uᶜ, committedpolicy
end

@printf "\nSaved committed policy to %s\n" solutionpath

## Plot committed policy
begin
    polfig = plot(mgrid, committedpolicy ./ taxfactor; xlabel = L"m", ylabel = L"Carbon tax USD per $\mathrm{CO}_2 \mathrm{e}$", c = :darkred, yguidefontcolor = :darkred, xlims = extrema(mgrid), label = false)
    plot!(twinx(polfig), mgrid, [e(a(τ, government, firm), firm) for τ in committedpolicy]; ylabel = L"e^c", c = :darkblue, yguidefontcolor = :darkblue, xlims = extrema(mgrid), label = false)
    savefig(polfig, joinpath(figurepath, "committed-policy.png"))

    polfig
end
