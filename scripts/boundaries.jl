## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots

import FastClosures: @closure
import UnPack: @unpack

import JLD2
import LinearAlgebra as LA
import SparseArrays as SA

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

solutionlabel = comittedsolutionlabel(firm, government, climate, mgrid)
solutionpath = joinpath(SIMPATH, "$solutionlabel.jld2")

JLD2.jldopen(solutionpath) do file
    @unpack uᶜ, committedpolicy, mgrid = committedfile
end 

## Lower boundary

## Upper boundary
