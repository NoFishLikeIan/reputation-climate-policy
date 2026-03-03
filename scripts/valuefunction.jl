## Load packages
using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack

using Optim, Roots

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background = :white)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/grid.jl")
includet("../src/optimal.jl")
includet("../src/utils.jl")
includet("../src/valuefunction.jl")

firm = Firm()
government = Government()

abatementdomain = (0., 1.2) .* firm.e₀
beliefdomain = (0., 1.)
stategrid = Grid((51, 50), (abatementdomain, beliefdomain));

investmentspace = (0., 0.1) .* firm.e₀
taxspace = (0., 20.)
controlgrid = Grid((61, 60), (investmentspace, taxspace));

const τᶜ = 10.

## Firm value
V = FirmValue(stategrid, controlgrid)

## Government value
S = Welfare(stategrid)