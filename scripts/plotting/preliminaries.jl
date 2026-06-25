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

plotpath = "figures/preliminaries"
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
    secondary = beliefscolors[:teal],
    reference = beliefscolors[:dark],
    guide = beliefscolors[:muted],
    series = [beliefscolors[:green], beliefscolors[:teal], beliefscolors[:olive], beliefscolors[:brown]],
)

begin
    damagefig = Plots.plot(
        mgrid,
        m -> d(m, climate);
        xlabel = L"Cumulative emissions $m_t$ [GtCO2e]",
        ylabel = "Output loss [% GDP / year]",
        c = preliminarycolors.primary,
        ylims = (0, Inf),
        xlims = extrema(mgrid),
        label = L"Damages $d(m)$",
        legend = :topleft,
        yaxis = (formatter = percentageformatter),
    )

    Plots.hline!(damagefig, [c(firm.e₀, firm)]; linestyle = :dash, c = preliminarycolors.reference, label = L"Net-zero abatement costs $c(e_0)$")
    Plots.plot!(damagefig, [m₀, m₀], [0., d(m₀, climate)]; c = preliminarycolors.guide, linestyle = :dot)
    Plots.plot!(damagefig, [0., m₀], [d(m₀, climate), d(m₀, climate)]; c = preliminarycolors.guide, linestyle = :dot)
    Plots.scatter!(damagefig, [m₀], [d(m₀, climate)]; c = preliminarycolors.primary, markerstrokewidth = 0)
    safesavefigure(damagefig, joinpath(plotpath, "damages.png"))
end

## Mac curve
agrid = range(0, firm.e₀, 501)

begin
    macfig = Plots.plot(
        agrid,
        a -> firm.ν * (firm.e₀ - a);
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
end

## Stranded assets costs
τgrid = range(0, 10τ₀, 501)

begin
    transitionfig = Plots.plot(
        xlabel = L"Abatement $a_t$ [GtCO2e / year]",
        ylabel = L"Transition-loss loading",
        xlims = extrema(agrid),
        ylims = (0, Inf),
        legend = :topright,
    )

    Plots.plot!(
        transitionfig,
        agrid,
        a -> R(a, government, firm);
        c = preliminarycolors.primary,
        label = L"Residual exposure $R(a)$",
    )
    Plots.plot!(
        transitionfig,
        agrid,
        a -> A(a, government, firm);
        c = preliminarycolors.secondary,
        label = L"Accelerated retirement $A(a)$",
    )
    Plots.plot!(
        transitionfig,
        agrid,
        a -> R(a, government, firm) + A(a, government, firm);
        c = preliminarycolors.reference,
        linestyle = :dash,
        label = L"$R(a)+A(a)$",
    )
    Plots.vline!(
        transitionfig,
        [firm.ω * firm.e₀];
        c = preliminarycolors.guide,
        linestyle = :dot,
        label = L"Free abatement $\omega e_0$",
    )

    safesavefigure(transitionfig, joinpath(plotpath, "transition-loss-components.png"))

    transitionfig
end

begin
    lfig = Plots.plot(
        xlabel = L"Carbon tax $\tau$ [trUSD / GtCO2e]",
        xlims = extrema(τgrid),
        ylabel = L"Stranded assets loss $l(a, \tau) / y_0$ [% GDP / year]",
        legend_title = L"Abatement $a$ [GtCO2e / year]",
        legend = :topleft,
        yaxis = (formatter = percentageformatter),
        ylims = (0, Inf),
    )

    for (i, a) in enumerate([a₀, 0.5e₀, 0.8e₀, e₀])
        Plots.plot!(
            lfig,
            τgrid,
            τ -> l(τ, a, government, firm) / government.y₀;
            c = preliminarycolors.series[i],
            label = @sprintf("%.1f", a),
        )
    end

    safesavefigure(lfig, joinpath(plotpath, "stranded-assets-costs.png"))

    lfig
end

preliminaryfig = Plots.plot(damagefig, macfig, transitionfig, lfig; layout = (2, 2), size = (980, 720), margins = 6Plots.mm)
safesavefigure(preliminaryfig, joinpath(plotpath, "preliminaries.png"))

preliminaryfig
