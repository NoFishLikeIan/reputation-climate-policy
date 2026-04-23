## Modules
using Revise
using UnPack

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using FastClosures
using Optim, NonlinearSolve
using StaticArrays
using LinearAlgebra

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

ns = (51, 101, 101)
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
### Initialise firm
firmvalue = FirmValue(exantegrid, pricespace)
### Initialise welfare
welfare = ValueFunction(exantegrid)

## Iteration
φlims = (0., 1.)
σpath = range(3signal.σ, signal.σ, length = 4) # Homotopy path
algorithm = LimitedMemoryBroyden(threshold = 6)

homotopynonlinear!(firmvalue, welfare, τᶜ, exantegrid, pricespace, firm, government, signal; σpath, algorithm, maxiter = 50, valtol = 1e-4, poltol = 1e-4, φlims, τlims, verbose = 1)

## Analyse
function plotoverspace(V::TV; kwargs...) where TV <: AbstractMatrix
    contourf(reputationspace, abatementspace, V; xlabel = L"Reputation $z$", ylabel = L"Abatement $a$", linewidth = 0.5, c = :Reds, clims = (0, Inf), kwargs...)
end
