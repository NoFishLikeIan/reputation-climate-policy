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

plotpath = "figures/preliminaries/cumulative"
mkpath(plotpath)

includet("utils.jl")
includet("colors.jl")
publicationdefaults!()

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/primitives/climate.jl")
includet("../../src/agents/government.jl")
includet("../../src/utils/arguments.jl")

firm, government, signal, climate = initmodels()

## Welfare costs
Δm = 100firm.e₀ # 50 years without abatement
mgrid = range(0., m₀ + Δm, 501);
percentageformatter = @closure x -> @sprintf "%.2f%%" 100x
preliminarycolors = (
    primary = beliefscolors[:green],
    secondary = beliefscolors[:sage],
    reference = beliefscolors[:dark],
    guide = beliefscolors[:muted],
    series = beliefspalette(4),
);

begin
    damagecolor = beliefscolors[:red]

    damagefig = Plots.plot(
        mgrid,
        m -> d(m, climate);
        xlabel = L"Cumulative emissions $m_t$ [GtCO2e]",
        ylabel = "Output loss [% GDP / year]",
        c = damagecolor,
        ylims = (0, Inf),
        xlims = extrema(mgrid),
        label = L"Damages $d(m)$",
        yaxis = (formatter = percentageformatter),
    )

    Plots.plot!(damagefig, [m₀, m₀], [0., d(m₀, climate)]; c = preliminarycolors.guide, linestyle = :dot)
    Plots.plot!(damagefig, [0., m₀], [d(m₀, climate), d(m₀, climate)]; c = preliminarycolors.guide, linestyle = :dot)
    Plots.scatter!(damagefig, [m₀], [d(m₀, climate)]; c = damagecolor, markerstrokewidth = 0)
    safesavefigure(damagefig, joinpath(plotpath, "damages.png"))

    damagefig
end

## Mac curve
agrid = range(0, firm.e₀, 501)

begin
    macfig = Plots.plot(
        agrid,
        a -> c(firm.e₀ - a, firm) / government.y₀;
        xlabel = L"Abatement $a_t$ [GtCO2e / year]",
        ylabel = "Output loss [% GDP / year]",
        c = preliminarycolors.secondary,
        label = L"Marginal abatement cost $c'(a)$",
        legend = :topright,
        yaxis = (formatter = percentageformatter),
        ylims = (0, Inf),
        xlims = extrema(agrid),
    )
    safesavefigure(macfig, joinpath(plotpath, "marginal-abatement-costs.png"))
    macfig
end
