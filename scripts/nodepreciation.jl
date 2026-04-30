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

## Imports
includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/primitives/grid.jl")

includet("../src/solve/boundary.jl")
includet("../src/solve/valuefunction.jl")
includet("../src/solve/pfi.jl")
includet("../src/solve/equilibrium.jl")

## Setup
firm = Firm(δ = 0.)
government = Government()

ns = (101, 51, 51)

σestimated = 1.1;
signal = Signal(1., σestimated * scctotax, ns[3])

τᶜ = optimize(τᶜ -> w̄(e₀, τᶜ, firm, government, signal), 0., 1., Optim.Brent()).minimizer
τlims = (0., 2τᶜ)
ξmin, ξmax = extrema(signal.space[1])

emissionsspace = range(0, e₀, ns[1])
Δzmax = maximum(abs, (
    ℓ(realisedprice(ξ, τ, signal), τ, τᶜ, signal)
    for ξ in (ξmin, ξmax), τ in τlims
))
zmax = max(8.0, 6Δzmax)
reputationspace = collect(range(-zmax, zmax, length = ns[2]))
exantegrid = Grid((emissionsspace, reputationspace))

qmin = minimum(realisedprice(ξ, τ, signal) for ξ in (ξmin, ξmax), τ in τlims)
qmax = maximum(realisedprice(ξ, τ, signal) for ξ in (ξmin, ξmax), τ in τlims)
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

## Analyse
using Plots, LaTeXStrings
Plots.default(label = false, dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm, linewidth = 2.5)

function plotoverspace(V::TV; kwargs...) where TV <: AbstractMatrix
    heatmap(reputationspace, emissionsspace, V; xlabel = L"Reputation $z$", ylabel = L"Emissions $e$", linewidth = 0.5, c = :Reds, clims = (0, Inf), kwargs...)
end
