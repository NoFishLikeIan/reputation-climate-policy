## Setup
using Revise

import Printf
import JLD2
import SciMLBase
import FastInterpolations as Itp

import LinearSolve
import SparseArrays
import StaticArrays as SA
import UnPack: @unpack

import CairoMakie
import Colors
import LaTeXStrings: @L_str

plotpath = "figures/preliminaries"
if !ispath(plotpath) mkpath(plotpath) end

includet("colours.jl")

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/primitives/climate.jl")

includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/dynamics/state.jl")
includet("../../src/dynamics/belief.jl")
includet("../../src/dynamics/firm.jl")
includet("../../src/dynamics/government.jl")

includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("../../src/solve/government/committed.jl")
includet("../../src/solve/government/noncommitted.jl")

includet("../../src/dynamics/simulation.jl")

firm, government, signal, climate = initmodels()

## Welfare costs
Δm = 100firm.e₀ # 50 years without abatement
mgrid = range(0., m₀ + Δm, 501);
percentageformatter = x -> Printf.@sprintf "%.2f%%" 100x

begin
    damagevalues = map(m -> d(m, climate), mgrid)
    initialdamage = d(m₀, climate)

    damagefig = CairoMakie.Figure(size = (600, 400))
    damageaxis = CairoMakie.Axis(
        damagefig[1, 1];
        xlabel = L"Cumulative emissions $m_t$ [GtCO2e]",
        ylabel = "Output loss [% GDP / year]",
        limits = (extrema(mgrid), (0, nothing)),
        ytickformat = values -> [Printf.@sprintf "%.0f%%" 100x for x in values],
        yticks = 0:0.01:0.05
    )

    CairoMakie.lines!(damageaxis, mgrid, damagevalues; color = defaultpalette[:damages], label = L"Damages $d(m)$")
    CairoMakie.lines!(damageaxis, [m₀, m₀], [0, initialdamage]; color = defaultpalette[:guide], linestyle = :dot)
    CairoMakie.lines!(damageaxis, [0, m₀], [initialdamage, initialdamage]; color = defaultpalette[:guide], linestyle = :dot)
    CairoMakie.scatter!(damageaxis, [m₀], [initialdamage]; color = defaultpalette[:damages], strokewidth = 0)
    CairoMakie.axislegend(damageaxis; position = :lt)
    CairoMakie.save(joinpath(plotpath, "damages.png"), damagefig)

    damagefig
end

## Mac curve
agrid = range(0, firm.e₀, 501)

begin
    macvalues = map(a -> c(firm.e₀ - a, firm) / government.y₀, agrid)

    macfig = CairoMakie.Figure(size = (600, 400))
    macaxis = CairoMakie.Axis(
        macfig[1, 1];
        xlabel = L"Abatement $a_t$ [GtCO2e / year]",
        ylabel = L"Output loss [% GDP / year] $$",
        limits = (extrema(agrid), (0, nothing)),
        ytickformat = values -> [Printf.@sprintf "%.1f%%" 100x for x in values],
        yticks = (0:0.5:2) ./ 100
    )

    CairoMakie.lines!(macaxis, agrid, macvalues; color = defaultpalette[:mac], label = L"Marginal abatement cost $c'(a)$")
    CairoMakie.axislegend(macaxis; position = :rt)
    CairoMakie.save(joinpath(plotpath, "marginal-abatement-costs.png"), macfig)

    macfig
end
