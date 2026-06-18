## Setup
using Revise, BenchmarkTools
using Printf


import FastClosures: @closure
import UnPack: @unpack, @pack!

import FastInterpolations as Itp
import BatchSolve
import JLD2
import SpecialFunctions as SF
import LinearAlgebra as LA
import Optim
import SparseArrays as SA

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/primitives/climate.jl")
includet("../src/agents/government.jl")
includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/committedvalue.jl")
includet("../src/solve/boundaries.jl")
includet("../src/solve/interior.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm, government, signal, climate = initmodels()

filename = solutionlabel(climate, government, firm, signal)
solutionpath = joinpath(SIMPATH, "$filename.jld2")
figurepath = joinpath("figures", filename)
mkpath(figurepath)

solutionfile = JLD2.jldopen(solutionpath)
@unpack committedpolicy, mgrid = solutionfile["committed"]
@unpack ū = solutionfile["upper"]
close(solutionfile)

## Boundaries
committedmgrid = mgrid
τᶜ = Itp.linear_interp(committedmgrid, committedpolicy; extrap = Itp.ClampExtrap())
ūitp = Itp.linear_interp(committedmgrid, ū; extrap = Itp.ClampExtrap())

## Interior
mgrid = range(first(committedmgrid), last(committedmgrid), 51)
φgrid = range(0., 1., 51)
u̲grid = [u̲(m, climate, government, firm) for m in mgrid]
ūgrid = [ūitp(m) for m in mgrid]
u = initialinteriorvalue(φgrid, mgrid, u̲grid, ūgrid)

_, interiorpolicy, (i, abserror, relerror) = solveinteriorhjb!(u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal, climate, government, firm; maxiters = 10_000, verbose = 1, abstol = 1e-6, reltol = 1e-4, Δt⁻¹ = 10.)

JLD2.jldopen(solutionpath, "a+") do file
    solution = JLD2.Group(file, "interior")
    @pack! solution = φgrid, mgrid, u, interiorpolicy, u̲grid, ūgrid
end

@printf "\nSaved interior solution to %s\n" solutionpath

## Plot interior
if isinteractive()
    using LaTeXStrings, Plots

    Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = "#FAFAFA", size = 400 .* (√2, 1))

    valuefig = heatmap(mgrid, φgrid, u; xlabel = L"m", ylabel = L"\phi", title = L"u(\phi,m)")
    policyfig = heatmap(mgrid, φgrid, interiorpolicy; xlabel = L"m", ylabel = L"\phi", title = L"\tau^*(\phi,m)")

    interiorfig = plot(valuefig, policyfig; size = 600 .* (2√2, 1))

    savefig(valuefig, joinpath(figurepath, "interior-value.png"))
    savefig(policyfig, joinpath(figurepath, "interior-policy.png"))

    interiorfig
end
