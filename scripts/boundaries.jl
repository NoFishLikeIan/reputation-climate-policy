## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = :transparent, size = 400 .* (√2, 1))

import FastClosures: @closure
import UnPack: @unpack

import JLD2
import LinearAlgebra as LA
import SparseArrays as SA
import FastInterpolations as Itp
import Roots
import DifferentialEquations as DE
import SpecialFunctions as SF

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/committedvalue.jl")
includet("../src/solve/boundaries.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm = Firm()
government = Government()
signal = Signal()
climate = Climate()

filename = solutionlabel(climate, government, firm, signal)
solutionpath = joinpath(SIMPATH, "$filename.jld2")

solutionfile = JLD2.jldopen(solutionpath)
@unpack uᶜ, committedpolicy, mgrid = solutionfile["committed"]
close(solutionfile) 

## Upper boundary
τᶜ = Itp.linear_interp(mgrid, committedpolicy; extrap = Itp.ClampExtrap())
m̄ = mgrid[findfirst(m -> e₀ ≈ a(τᶜ(m), firm), mgrid)]

parameters = τᶜ, m̄, climate, government, firm
∂ₘΔu₀ = -government.y₀ * d′(m̄, climate)
Δu₀ = w(m̄, 0., firm.e₀, climate, government, firm)

upperboundaryprob = DE.DAEProblem(boundaryupperreversedae, ∂ₘΔu₀, Δu₀, (0., m̄ .- mgrid[1]), parameters)

upperboundarysol = DE.solve(upperboundaryprob)
ū = [upperboundarysol(m̄ - m) for m in mgrid]

JLD2.jldopen(solutionpath, "a+") do file
    solution = JLD2.Group(file, "upper")
    @pack! solution = ū, mgrid
end

## Plot boundaries
begin
    ufig = plot(mgrid, uᶜ; label = L"Comitted solution $u^c$", ylabel = "[tUSD]", xlabel = L"Cumulative emissions $m$ [GtCO2]", xlims = extrema(mgrid))

    plot!(mgrid, ū; label = L"Upper boundary $\bar{u}(m)$")
    plot!(mgrid, m -> u̲(m, climate, government, firm); label = L"Lower boundary $\underbar{u}(m)$")
end