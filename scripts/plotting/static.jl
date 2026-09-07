using Revise

import DotEnv
DotEnv.load!()

import JLD2
import SciMLBase
import FastInterpolations as Itp
import LinearSolve
import SparseArrays
import StaticArrays as SA
import Statistics
import OrdinaryDiffEq as ODE
import UnPack: @unpack
import LaTeXStrings: @L_str
import Printf
import CairoMakie
import Colors

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

includet("publication.jl")
includet("colours.jl")

CairoMakie.set_theme!(publicationtheme)

## Load problem
firm, government, signal, climate = initmodels()
taxmethod = OneShotTax()

filename = solutionfilename(climate, government, firm)
datapath = get(ENV, "DATAPATH", "data")
plotpath = get(ENV, "PLOTPATH", "figures")
solpath = joinpath(datapath, "solutions", filename)
isfile(solpath) || error("File $solpath not found.")

trajectory, committedtaxes, committedtime = JLD2.jldopen(solpath, "r") do file
    file["trajectory"], file["taxes"], file["time"]
end

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)
activecommittedtax = Itp.linear_interp(
    committedtime,
    committedtaxes;
    extrap = Itp.ClampExtrap(),
)
τᶜ = CommittedTaxPath(
    activecommittedtax,
    activeterminal,
    terminal,
    terminalabatement,
    firm,
    government,
)

figurepath = joinpath(
    plotpath,
    splitext(filename)[1],
    "comparative-statics",
    taxmethodlabel(taxmethod),
)
ispath(figurepath) || mkpath(figurepath)

## Signal-volatility comparison
comparisonσs = (signal.σ, 2signal.σ)
function comparisonslice(σ)
    comparisonsignal = Signal(; ϵ = signal.ϵ, σ)
    solutionkey = uncommittedsolutionkey(comparisonsignal, taxmethod)
    solution, grid, savedtaxmethod = JLD2.jldopen(solpath, "r") do file
        haskey(file, solutionkey) || error(
            "Uncommitted solution $solutionkey not found in $solpath.",
        )
        file["$solutionkey/solution"], file["$solutionkey/grid"], file["$solutionkey/taxmethod"]
    end
    parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, comparisonsignal, climate, savedtaxmethod)
    policies = noncommittedpoliciesattime(solution, parameters, 0.0)

    mindex = argmin(abs.(grid.mgrid .- climate.m₀))
    aindex = argmin(abs.(grid.agrid .- firm.a₀))
    isapprox(grid.mgrid[mindex], climate.m₀) || error(
        "The cumulative-emissions grid does not contain m₀ = $(climate.m₀).",
    )
    isapprox(grid.agrid[aindex], firm.a₀) || error(
        "The abatement grid does not contain a₀ = $(firm.a₀).",
    )

    # The second node uses a backward difference against the φ = 0 boundary
    # and can inherit a visible boundary artifact.
    φindices = (firstindex(grid.φgrid) + 2):(lastindex(grid.φgrid) - 1)
    φvalues = collect(grid.φgrid[φindices])
    taxvalues = collect(policies.tax[φindices, mindex, aindex])
    expectedtaxvalues = collect(policies.expectedtax[φindices, mindex, aindex])
    s = noncommittedreversetime(0.0, parameters)
    values = noncommittedvalues(solution(s), parameters)
    welfarecost = 1_000 .* collect(values.W[φindices, mindex, aindex])
    committedtax = τᶜ(0.0)
    signalprecision = χ.(taxvalues, committedtax, Ref(comparisonsignal))
    reputationloss = -100 .* beliefdrift.(signalprecision, φvalues)

    return (; σ, φvalues, taxratio = taxvalues ./ committedtax, expectedtaxratio = expectedtaxvalues ./ committedtax, reputationloss, welfarecost)
end
comparisons = map(comparisonslice, comparisonσs)

## Plot
panelwidth = publicationdefault(:panelwidth)
panelheight = publicationdefault(:panelheight) + 40
comparisoncolors = (
    defaultpalette[:committed],
    defaultpalette[:damages],
)
percentticks = 0:0.25:1
beliefxticks = 0:0.2:1

begin
    volatilityfig = CairoMakie.Figure(size = (3 * panelwidth, panelheight))

    taxaxis = CairoMakie.Axis(
        volatilityfig[1, 1];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Implemented tax $\tau / \tau^{\mathrm{c}}$",
        title = "(a) Government policy",
        limits = ((0, 1), (0, 1.02)),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = percentticks,
        ytickformat = percenttickformat,
    )
    expectedtaxaxis = CairoMakie.Axis(
        volatilityfig[1, 2];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Expected tax $\tau^e / \tau^{\mathrm{c}}$",
        title = "(b) Firms' incentives",
        limits = ((0, 1), (0, 1.02)),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = percentticks,
        ytickformat = percenttickformat,
    )
    reputationaxis = CairoMakie.Axis(
        volatilityfig[1, 3];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Expected loss $-\mu_\phi$ [pp/year]",
        title = "(c) Reputation dynamics",
        limits = ((0, 1), (0, nothing)),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )

    for (comparison, color) in zip(
        comparisons,
        comparisoncolors,
    )
        label = L"$\sigma = %$(comparison.σ)$"
        lineoptions = (;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )

        CairoMakie.lines!(
            taxaxis,
            comparison.φvalues,
            comparison.taxratio;
            lineoptions...,
        )
        CairoMakie.lines!(
            expectedtaxaxis,
            comparison.φvalues,
            comparison.expectedtaxratio;
            lineoptions...,
        )
        CairoMakie.lines!(
            reputationaxis,
            comparison.φvalues,
            comparison.reputationloss;
            lineoptions...,
        )
    end

    CairoMakie.Legend(
        volatilityfig[0, 1:3],
        taxaxis;
        orientation = :horizontal,
        tellwidth = false,
        title = "Signal volatility",
    )

    CairoMakie.linkxaxes!(taxaxis, expectedtaxaxis, reputationaxis)
    CairoMakie.Label(
        volatilityfig[2, 1:3],
        L"Evaluated at $t = 0$, $m = m_0$, and $a = a_0$";
        fontsize = 12,
        color = defaultpalette[:guide],
    )

    savepublicationfigure(
        joinpath(figurepath, "signal-volatility"),
        volatilityfig,
    )

    println("Saved signal-volatility comparison in ", figurepath)

    volatilityfig
end

## Welfare comparison
referencebeliefs = first(comparisons).φvalues
all(comparison -> comparison.φvalues == referencebeliefs, comparisons) || error(
    "The signal-volatility solutions use different belief grids.",
)
welfaredifference = last(comparisons).welfarecost .- first(comparisons).welfarecost

begin
    welfarefig = CairoMakie.Figure(size = (2 * panelwidth, panelheight))

    welfareaxis = CairoMakie.Axis(
        welfarefig[1, 1];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Welfare costs $u$ [bn USD]",
        title = "(a) Welfare-cost functions",
        limits = ((0, 1), nothing),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )
    differenceaxis = CairoMakie.Axis(
        welfarefig[1, 2];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Change in costs $u_{\sigma=%$(comparisonσs[2])} - u_{\sigma=%$(comparisonσs[1])}$ [bn USD]",
        title = "(b) Effect of higher volatility",
        limits = ((0, 1), nothing),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )

    for (comparison, color) in zip(
        comparisons,
        comparisoncolors,
    )
        CairoMakie.lines!(
            welfareaxis,
            comparison.φvalues,
            comparison.welfarecost;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label = L"$\sigma = %$(comparison.σ)$",
        )
    end

    CairoMakie.hlines!(
        differenceaxis,
        [0.0];
        color = defaultpalette[:guide],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )
    CairoMakie.band!(
        differenceaxis,
        referencebeliefs,
        zero.(welfaredifference),
        welfaredifference;
        color = (defaultpalette[:damages], 0.15),
    )
    CairoMakie.lines!(
        differenceaxis,
        referencebeliefs,
        welfaredifference;
        color = defaultpalette[:damages],
        linewidth = publicationdefault(:medianlinewidth),
    )

    CairoMakie.Legend(
        welfarefig[0, 1:2],
        welfareaxis;
        orientation = :horizontal,
        tellwidth = false,
        title = "Signal volatility",
    )
    CairoMakie.linkxaxes!(welfareaxis, differenceaxis)
    CairoMakie.Label(
        welfarefig[2, 1:2],
        L"Evaluated at $t = 0$, $m = m_0$, and $a = a_0$";
        fontsize = 12,
        color = defaultpalette[:guide],
    )

    savepublicationfigure(
        joinpath(figurepath, "welfare-volatility"),
        welfarefig,
    )

    println("Saved welfare comparison in ", figurepath)

    welfarefig
end
