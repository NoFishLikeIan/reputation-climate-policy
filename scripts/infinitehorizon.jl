## Modules
using Revise
using UnPack

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using FastClosures
using Optim, Optimization, SimpleOptimization
using StaticArrays

using Printf

using Plots, LaTeXStrings
Plots.default(label = false, dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm, linewidth = 2.5)

## Imports
includet("../src/constants.jl")
includet("../src/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/grid.jl")
includet("../src/valuefunction.jl")
includet("../src/boundary.jl")
includet("../src/pfi.jl")

## Setup
firm = Firm()
government = Government()

ns = (51, 101, 101)
signal = Signal(1., 0.2, ns[3])
a₀ = 0.
τᶜ = optimize(τᶜ -> w̄(a₀, τᶜ, firm, government, signal), 0., 1., Brent()).minimizer
aᵇ = blissabatement(τᶜ, firm, signal)

abatementspace = range(0, 1.25aᵇ, ns[1])
reputationspace = gausshermite(ns[2])[1]
grid = Grid((abatementspace, reputationspace))

qmin = realisedprice(first(signal.space[1]), 0.0, signal)
qmax = realisedprice(last(signal.space[1]), 2τᶜ, signal)
pricespace = collect(range(qmin, qmax, length = ns[3]))


## Value Function
### Initialise firm
firmvalue = FirmValue(grid, pricespace)
### Initialise welfare
welfare = ValueFunction(grid)

## Iteration
firmparams = Dict(:maxiter => 1_000, :valtol => 1e-3, :poltol => 1e-1)
welfareparams = Dict(:maxiter => 1_000, :valtol => 1e-3, :poltol => 1e-1)

nestedpfi!(firmvalue, welfare, τᶜ, grid, pricespace, firm, government, signal; maxiter = 50, valtol = 1e-4, poltol = 1e-4, verbose = 2, firmparams = firmparams, welfareparams = welfareparams)

## Analyse
function plotoverspace(V::TV; kwargs...) where TV <: AbstractMatrix
    contourf(reputationspace, abatementspace, V; xlabel = L"Reputation $z$", ylabel = L"Abatement $a$", linewidth = 0.5, c = :Reds, clims = (0, Inf), kwargs...)
end