## Setup
using Revise, BenchmarkTools
using Printf
using LaTeXStrings, Colors

import FastClosures: @closure
import UnPack: @unpack

import JLD2
import LinearAlgebra as LA
import SparseArrays as SA

import Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = :transparent, size = 400 .* (√2, 1))

plotpath = "papers/figures/preliminaries"
if !isdir(plotpath) mkdir(plotpath) end

includet("colors.jl")

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/primitives/climate.jl")
includet("../../src/agents/government.jl")

firm = Firm()
government = Government()
signal = Signal()
climate = Climate()

## Welfare costs
Δm = 100firm.e₀ # 50 years without abatement
mgrid = range(0., m₀ + Δm, 501);
percentageformatter = x -> @sprintf "%.2f%%" 100x

begin
    damagefig = Plots.plot(mgrid, m -> d(m, climate); xlabel = L"Cumulative emissions $m_t$ [GtCO2e]", ylabel = "Output loss [% GDP / year]", c = :darkred, ylims = (0, Inf), xlims = extrema(mgrid), label = L"Damages $d(m)$", yaxis = (formatter = percentageformatter))

    Plots.hline!(damagefig, m -> c(firm.e₀, firm); linestyle = :dash, c = :black, label = L"Net-zero abatemnet costs $c(e_0)$")
    Plots.plot!(damagefig, [m₀, m₀], [0., d(m₀, climate)], c = :darkred, linestyle = :dot)
    Plots.plot!(damagefig, [0., m₀], [d(m₀, climate), d(m₀, climate)], c = :darkred, linestyle = :dot)
    Plots.scatter!(damagefig, [m₀], [d(m₀, climate)], c = :darkred, markerstrokewidth = 0)
end

## Mac curve
agrid = range(0, e₀, 501)

begin
    mac = Plots.plot(agrid, a -> firm.ν * (firm.e₀ - a); xlabel = L"Emissions $e_t$", ylabel = "Output loss [% GDP / year]", c = :darkred, yaxis = (formatter = percentageformatter), ylims = (0, Inf))
end
