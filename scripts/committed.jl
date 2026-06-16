## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots

import FastClosures: @closure
import UnPack: @unpack

import JLD2
import LinearAlgebra as LA
import SparseArrays as SA

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/committedvalue.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm = Firm()
government = Government()
signal = Signal()
climate = Climate()

Δm = 100firm.e₀ # 50 years without abatement
mgrid = range(m₀, m₀ + Δm, 501);

## Initialise value function problem
uᶜ₀ = [w(m, 0.01, a(0.01, firm), climate, government, firm) for m in mgrid]
uᶜ = copy(uᶜ₀)

maxiters = 100_000
verbose = 1
abstol = 1e-9
reltol = 1e-6
Δt⁻¹ = 100.

_, (i, abserror, relerror) = solvehjb!(uᶜ, mgrid, climate, government, firm; maxiters, verbose, abstol, reltol, Δt⁻¹)

comittedpolicy = computeglobalpolicy(uᶜ, mgrid, government, firm)
solutionlabel = comittedsolutionlabel(firm, government, climate)
solutionpath = joinpath(SIMPATH, "$solutionlabel.jld2")

JLD2.jldsave(solutionpath;solutionlabel, uᶜ, comittedpolicy, mgrid)

@printf "\nSaved committed policy to %s\n" solutionpath


## Plot committed policy
begin
    polfig = plot(mgrid, comittedpolicy; xlabel = L"m", ylabel = L"\tau^c", c = :darkred, yguidefontcolor = :darkred, xlims = extrema(mgrid), label = false)
    plot!(twinx(polfig), mgrid, e.(a.(comittedpolicy, Ref(firm)), Ref(firm)); ylabel = L"a^c", c = :darkblue, yguidefontcolor = :darkblue, xlims = extrema(mgrid), label = false)
end
