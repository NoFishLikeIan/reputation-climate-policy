## Modules
using Revise
using UnPack

using StaticArrays
using FastInterpolations, FastGaussQuadrature
using FastClosures
using LinearAlgebra
using LogExpFunctions
using Optim

using Printf

using Plots, LaTeXStrings
Plots.default(label = false, dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm, linewidth = 2.5)

## Imports
includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/primitives/grid.jl")

includet("../src/solve/valuefunction.jl")
includet("../src/solve/boundary.jl")
includet("../src/solve/pfi.jl")
includet("../src/solve/equilibrium.jl")

## Setup
firm = Firm(δ = 0.)
government = Government()

ns = (101, 51, 51)
signal = Signal(1., 1.1 * scctotax, ns[3])

τᶜ = optimize(τᶜ -> w̄(a₀, τᶜ, firm, government, signal), 0., 1., Optim.Brent()).minimizer
aᵘ = abatementgridupper(τᶜ, firm, signal)
τlims = (0., 2τᶜ)
ξmin, ξmax = extrema(signal.space[1])

abatementspace = range(0, aᵘ, ns[1])
Δzmax = maximum(abs, (
    ℓ(realisedprice(ξ, τ, signalᵢ), τ, τᶜ, signalᵢ)
    for signalᵢ in homotopysignals, ξ in (ξmin, ξmax), τ in τlims
))
zmax = max(8.0, 6Δzmax)
reputationspace = collect(range(-zmax, zmax, length = ns[2]))
exantegrid = Grid((abatementspace, reputationspace))

qmin = minimum(realisedprice(ξ, τ, signalᵢ) for signalᵢ in homotopysignals, ξ in (ξmin, ξmax), τ in τlims)
qmax = maximum(realisedprice(ξ, τ, signalᵢ) for signalᵢ in homotopysignals, ξ in (ξmin, ξmax), τ in τlims)
pricespace = collect(range(qmin, qmax, length = ns[3]))


## Value Function
firmvalue = FirmValue(exantegrid, pricespace)
welfare = ValueFunction(exantegrid)

## Iteration
iteration, firmvalue, welfare = steadypolicies!(
    firmvalue,
    welfare,
    τᶜ,
    exantegrid,
    pricespace,
    firm,
    government,
    signal;
    maxiter = 500,
    relax = 0.01,
    valtol = 1e-4,
    poltol = 1e-4,
    τlims,
    mimictol = 1e-8,
    mimicband = 1e-7,
    policyopttol = 1e-5,
    taxseparation = 0.01τᶜ,
    verbose = 2,
)
iterations = [iteration]

## Analyse
function plotoverspace(V::TV; kwargs...) where TV <: AbstractMatrix
    heatmap(reputationspace, abatementspace, V; xlabel = L"Reputation $z$", ylabel = L"Abatement $a$", linewidth = 0.5, c = :Reds, clims = (0, Inf), kwargs...)
end
