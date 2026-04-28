## Modules
using Revise
using UnPack

using StaticArrays
using FastInterpolations, FastGaussQuadrature
using FastClosures
using LinearAlgebra
using LogExpFunctions
using Optim, NonlinearSolve, SciMLBase

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
includet("../src/equilibrium.jl")

## Setup
firm = Firm()
government = Government()

ns = (21, 21, 21)
signal = Signal(1., 0.2, ns[3])
a₀ = 0.
τᶜ = optimize(τᶜ -> w̄(a₀, τᶜ, firm, government, signal), 0., 1., Optim.Brent()).minimizer
aᵇ = blissabatement(τᶜ, firm, signal)
τlims = (0., 2τᶜ)
ξmin, ξmax = extrema(signal.space[1])

abatementspace = range(0, 1.25aᵇ, ns[1])
Δzmax = maximum(abs, (ℓ(realisedprice(ξ, τ, signal), τ, τᶜ, signal) for ξ in (ξmin, ξmax), τ in τlims))
zmax = max(8.0, 6Δzmax)
reputationspace = collect(range(-zmax, zmax, length = ns[2]))
exantegrid = Grid((abatementspace, reputationspace))

qmin = minimum(realisedprice(ξ, τ, signal) for ξ in (ξmin, ξmax), τ in τlims)
qmax = maximum(realisedprice(ξ, τ, signal) for ξ in (ξmin, ξmax), τ in τlims)
pricespace = collect(range(qmin, qmax, length = ns[3]))


## Value Function
firmvalue = FirmValue(exantegrid, pricespace)
welfare = ValueFunction(exantegrid)

## Iteration
σpath = range(3signal.σ, signal.σ, length = 4) # Homotopy path
algorithm = LimitedMemoryBroyden(max_resets = 10)

solutions, firmvalue, welfare = homotopynonlinear!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; σpath, algorithm, maxiter = 50, valtol = 1e-4, poltol = 1e-4, verbose = 2)

## Analyse
function plotoverspace(V::TV; kwargs...) where TV <: AbstractMatrix
    contourf(reputationspace, abatementspace, V; xlabel = L"Reputation $z$", ylabel = L"Abatement $a$", linewidth = 0.5, c = :Reds, clims = (0, Inf), kwargs...)
end