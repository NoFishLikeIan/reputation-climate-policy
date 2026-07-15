## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
import JLD2

import Base.Threads
import FastClosures: @closure
import UnPack: @unpack, @pack!

import LinearAlgebra
import SparseArrays
import FastChebInterp

import Optim
import StaticArrays as SA
import StaticArraysCore
import LCPsolve

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/dynamics/state.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/utils.jl")
includet("../src/solve/firm/committed.jl")

const SIMPATH = joinpath("data", "solutions")

## Defaults
firm, government, signal, climate = initmodels()

## Chebyshev collocation grid
gridorder = (99, 100)
Δm = 80firm.e₀ # 80 years without abatement
lowerbound = SA.SVector(a₀, m₀)
upperbound = SA.SVector(firm.e₀, m₀ + Δm)
collocationpoints = FastChebInterp.chebpoints(gridorder, lowerbound, upperbound)


## Approximate the committed tax
τᶜinitguess = @closure u -> (u[2] / upperbound[2]) * defaultscc
τᶜ = FastChebInterp.chebinterp(τᶜinitguess.(collocationpoints), lowerbound, upperbound)

## Initialise value function problem
q = Matrix{Float64}(undef, size(collocationpoints));
I = Vector{Union{Nothing, Int}}(undef, gridorder[1] + 1);

solvefirmproblem!(q, I, τᶜ, collocationpoints, firm)
