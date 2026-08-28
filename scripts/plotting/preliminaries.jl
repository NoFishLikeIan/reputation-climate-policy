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

publicationtheme = CairoMakie.Theme(
    fontsize = 16,
    Axis = (;
        titlesize = 18,
        titlegap = 8,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xgridcolor = (:black, 0.08),
        ygridcolor = (:black, 0.08),
        topspinevisible = false,
        rightspinevisible = false,
    ),
    Legend = (;
        labelsize = 13,
        framevisible = false,
    ),
)
CairoMakie.set_theme!(publicationtheme)

singlepanelsize = (420, 300)
combinedfiguresize = (900, 620)

savepublicationfigure = function (basename, figure)
    CairoMakie.save("$basename.pdf", figure; pt_per_unit = 1)
    CairoMakie.save("$basename.png", figure; px_per_unit = 2)
end

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
denseyticks = CairoMakie.LinearTicks(8)
mainlinewidth = 3.5
guidelinewidth = 2.0

begin
    damagevalues = map(m -> d(m, climate), mgrid)
    initialdamage = d(m₀, climate)

    damagefig = CairoMakie.Figure(size = singlepanelsize)
    damageaxis = CairoMakie.Axis(
        damagefig[1, 1];
        xlabel = L"Cumulative emissions $m_t$ [GtCO2e]",
        ylabel = "Output loss [% GDP / year]",
        limits = (extrema(mgrid), (0, nothing)),
        ytickformat = values -> [Printf.@sprintf "%.1f%%" 100x for x in values],
        yticks = 0:0.005:0.05
    )

    CairoMakie.lines!(damageaxis, mgrid, damagevalues; color = defaultpalette[:damages], linewidth = mainlinewidth, label = L"Damages $d(m)$")
    CairoMakie.lines!(damageaxis, [m₀, m₀], [0, initialdamage]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
    CairoMakie.lines!(damageaxis, [0, m₀], [initialdamage, initialdamage]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
    CairoMakie.scatter!(damageaxis, [m₀], [initialdamage]; color = defaultpalette[:damages], strokewidth = 0)
    CairoMakie.axislegend(damageaxis; position = :lt)
    savepublicationfigure(joinpath(plotpath, "damages"), damagefig)

    damagefig
end

## Mac curve
agrid = range(0, firm.e₀, 501)

begin
    macvalues = map(a -> c(a, firm) / government.y₀, agrid)

    macfig = CairoMakie.Figure(size = singlepanelsize)
    macaxis = CairoMakie.Axis(
        macfig[1, 1];
        xlabel = L"Abatement $a_{i, t}$ [GtCO2e / year]",
        ylabel = L"Output loss [% GDP / year] $$",
        limits = (extrema(agrid), (0, nothing)),
        ytickformat = values -> [Printf.@sprintf "%.2f%%" 100x for x in values],
        yticks = (0:0.25:2) ./ 100
    )

    CairoMakie.lines!(macaxis, agrid, macvalues; color = defaultpalette[:mac], linewidth = mainlinewidth, label = L"Marginal abatement cost $c(a_{i, t})$")
    CairoMakie.axislegend(macaxis; position = :rt)
    savepublicationfigure(joinpath(plotpath, "marginal-abatement-costs"), macfig)

    macfig
end

## Calibration mechanisms
begin
    temperaturevalues = temperature.(mgrid, Ref(climate))
    initialtemperature = temperature(climate.m₀, climate)
    emissionsvalues = e.(agrid, Ref(firm))
    initialemissions = e(firm.a₀, firm)
    initialtaxdollars = τ₀ / taxfactor
    netzerotaxdollars = firm.r * c(firm.e₀, firm) / taxfactor
    taxgrid = range(
        0.0,
        1.05 * max(initialtaxdollars, netzerotaxdollars);
        length = 501,
    )
    longrunabatement = min.(
        taxfactor .* taxgrid ./ (firm.r * firm.κ),
        firm.e₀,
    )
    initialtaxabatement = min(
        τ₀ / (firm.r * firm.κ),
        firm.e₀,
    )
    initialabatementtaxdollars = firm.r * c(firm.a₀, firm) / taxfactor
    φgrid = range(0.0, 1.0; length = 501)
    calibrationtaxgap = τ₀
    calibrationχ = χ(zero(calibrationtaxgap), calibrationtaxgap, signal)
    beliefdriftvalues = beliefdrift.(calibrationχ, φgrid)
    beliefdiffusionvalues = beliefdiffusion.(calibrationχ, φgrid)

    plottemperaturecalibration! = function (position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Cumulative emissions $m$ [GtCO2e]",
            ylabel = "Temperature [°C]",
            title = L"Temperature response $T(m)$",
            limits = (extrema(mgrid), (0, nothing)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, mgrid, temperaturevalues; color = defaultpalette[:damages], linewidth = mainlinewidth, label = L"$T(m)=\zeta m$")
        CairoMakie.vlines!(axis, [climate.m₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.hlines!(axis, [initialtemperature]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.scatter!(axis, [climate.m₀], [initialtemperature]; color = defaultpalette[:damages], markersize = 14, strokewidth = 0, label = "Initial calibration")
        CairoMakie.axislegend(axis; position = :lt)

        axis
    end

    plotemissionscalibration! = function (position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Abatement $a$ [GtCO2e/year]",
            ylabel = "Emissions [GtCO2e/year]",
            title = L"Residual emissions $e(a)$",
            limits = (extrema(agrid), (0, 1.05 * firm.e₀)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, agrid, emissionsvalues; color = defaultpalette[:emissions], linewidth = mainlinewidth, label = L"$e(a)=e_0-a$")
        CairoMakie.vlines!(axis, [firm.a₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.hlines!(axis, [initialemissions]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.scatter!(axis, [firm.a₀], [initialemissions]; color = defaultpalette[:emissions], markersize = 14, strokewidth = 0, label = "Initial calibration")
        CairoMakie.scatter!(axis, [firm.e₀], [0.0]; color = defaultpalette[:abatement], markersize = 14, strokewidth = 0, label = "Net zero")
        CairoMakie.axislegend(axis; position = :rt)

        axis
    end

    plottaxabatementcalibration! = function (position)
        axis = CairoMakie.Axis(
            position;
            xlabel = "Carbon tax [USD/tCO2e]",
            ylabel = "Abatement [GtCO2e/year]",
            title = L"Long-run firm response $a^*(\tau)$",
            limits = (extrema(taxgrid), (0, 1.05 * firm.e₀)),
            yticks = denseyticks,
        )
        CairoMakie.lines!(axis, taxgrid, longrunabatement; color = defaultpalette[:committed], linewidth = mainlinewidth, label = L"$\tau=r_f c(a)$")
        CairoMakie.vlines!(axis, [initialtaxdollars]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.hlines!(axis, [firm.a₀, firm.e₀]; color = defaultpalette[:guide], linestyle = :dot, linewidth = guidelinewidth)
        CairoMakie.scatter!(axis, [initialtaxdollars], [initialtaxabatement]; color = defaultpalette[:abatement], markersize = 14, strokewidth = 0, label = L"Response to $\tau_0$")
        CairoMakie.scatter!(axis, [initialabatementtaxdollars], [firm.a₀]; color = defaultpalette[:emissions], markersize = 14, strokewidth = 0, label = L"Tax sustaining $a_0$")
        CairoMakie.text!(axis, last(taxgrid), firm.e₀; text = "Net zero", align = (:right, :bottom), offset = (0, 4), color = defaultpalette[:guide], fontsize = 14)
        CairoMakie.axislegend(axis; position = :lt)

        axis
    end

    plotbeliefdriftcalibration! = function (position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Belief $\phi$",
            ylabel = "Belief drift",
            title = Printf.@sprintf(
                "Belief drift for a %.0f USD/tCO2e tax gap",
                calibrationtaxgap / taxfactor,
            ),
            limits = ((0, 1), nothing),
            xticks = 0:0.1:1,
            yticks = denseyticks,
            ytickformat = values -> percentageformatter.(values),
        )
        CairoMakie.hlines!(axis, [0.0]; color = defaultpalette[:guide], linewidth = guidelinewidth)
        CairoMakie.lines!(axis, φgrid, beliefdriftvalues; color = defaultpalette[:damages], linewidth = mainlinewidth, label = L"$\mu_\phi(\phi)$")
        CairoMakie.axislegend(axis; position = :rb)

        axis
    end

    plotbeliefdiffusioncalibration! = function (position)
        axis = CairoMakie.Axis(
            position;
            xlabel = L"Belief $\phi$",
            ylabel = "Belief diffusion",
            title = Printf.@sprintf(
                "Belief diffusion for a %.0f USD/tCO2e tax gap",
                calibrationtaxgap / taxfactor,
            ),
            limits = ((0, 1), (0, nothing)),
            xticks = 0:0.1:1,
            yticks = denseyticks,
            ytickformat = values -> percentageformatter.(values),
        )
        CairoMakie.lines!(axis, φgrid, beliefdiffusionvalues; color = defaultpalette[:committed], linewidth = mainlinewidth, label = L"$\sigma_\phi(\phi)$")
        CairoMakie.axislegend(axis; position = :rt)

        axis
    end

    calibrationfig = CairoMakie.Figure(size = combinedfiguresize)
    CairoMakie.Label(
        calibrationfig[0, 1:3],
        "Calibration mechanisms";
        fontsize = 20,
        tellwidth = false,
    )
    plottemperaturecalibration!(calibrationfig[1, 1])
    plotemissionscalibration!(calibrationfig[1, 2])
    plottaxabatementcalibration!(calibrationfig[1, 3])
    beliefgrid = calibrationfig[2, 1:3] = CairoMakie.GridLayout()
    plotbeliefdriftcalibration!(beliefgrid[1, 1])
    plotbeliefdiffusioncalibration!(beliefgrid[1, 2])

    savepublicationfigure(
        joinpath(plotpath, "calibration-mechanisms"),
        calibrationfig,
    )

    individualpanels = (
        ("temperature-response", plottemperaturecalibration!),
        ("residual-emissions", plotemissionscalibration!),
        ("tax-abatement-response", plottaxabatementcalibration!),
        ("belief-drift", plotbeliefdriftcalibration!),
        ("belief-diffusion", plotbeliefdiffusioncalibration!),
    )
    for (filename, plotpanel!) in individualpanels
        panelfig = CairoMakie.Figure(size = singlepanelsize)
        plotpanel!(panelfig[1, 1])
        savepublicationfigure(joinpath(plotpath, filename), panelfig)
    end

    calibrationfig
end
