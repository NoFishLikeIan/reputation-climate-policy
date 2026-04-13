## Load packages
using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack
using Printf

using Optim, Roots

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/utils.jl")
includet("../src/nc.jl")

## Plotting utilities
using Plots, LaTeXStrings
using Measures

Plots.default(linewidth = 2, dpi = 180, label = false, background = :white)
plotpath = "figures/aps"; if !ispath(plotpath) mkpath(plotpath) end

## Model
firm = Firm()
government = Government()

## NC equilibrium
n = 101
A = range(0, 1.2firm.e₀, n)

m = 21
T = range(0, 2, m)

Vⁿ = Value(ones(n, n), zeros(n, n))
Sⁿ = Value(ones(n), zeros(n))

## Steady state
steadystate!(Sⁿ, Vⁿ, A, T, firm, government; verbose = 1, totaliter = 1000, ρ = 0.01)