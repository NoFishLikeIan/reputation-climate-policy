using Revise

import DotEnv; DotEnv.load!()

import JLD2
import UnPack: @unpack
import LaTeXStrings: @L_str
import Printf

import SciMLBase
import FastInterpolations as Itp
import LinearSolve
import SparseArrays
import StaticArrays as SA
import Statistics
import StochasticDiffEq as SDE
import StochasticDiffEqImplicit as SDEImplicit
import OrdinaryDiffEq as ODE

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

## Plotting
import CairoMakie
import Colors

includet("publication.jl")
includet("colours.jl")
includet("simulationplots.jl")

CairoMakie.set_theme!(publicationtheme)

## Load problem
## Setup import
firm, government, signal, climate = initmodels()

taxmethod = OneShotTax()
filename = solutionfilename(climate, government, firm)
solpath = joinpath("data", "solutions", filename)
if !isfile(solpath) throw("File $solpath not found.") end

solution, grid, taxmethod, trajectory, committedtaxes, committedtime = JLD2.jldopen(solpath, "r") do file
    solutionkey = uncommittedsolutionkey(signal, taxmethod)

    if !haskey(file, solutionkey) error("Uncommitted solution $solutionkey not found in $solpath.") end

    (
        file["$solutionkey/solution"],
        file["$solutionkey/grid"],
        file["$solutionkey/taxmethod"],
        file["trajectory"],
        file["taxes"],
        file["time"],
    )
end

activeterminal = committedtime[end]
terminalabatement = trajectory[end][2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)

activecommittedtax = Itp.linear_interp(committedtime, committedtaxes; extrap = Itp.ClampExtrap())
τᶜ = CommittedTaxPath(activecommittedtax, activeterminal, terminal, terminalabatement, firm, government)

models = (firm, government, signal, climate)
parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate, taxmethod)
policies = constructpolicies(solution, parameters, grid)

## Simulate path
x₀ = SA.SVector(0., climate.m₀, firm.a₀)
dynamicparameters = (policies, τᶜ, terminal, models)
horizonsimulation = terminal

dynamicfn = SDE.SDEFunction{false}(logdynamicdrift, logdynamicnoise)
dynamicprob = SDE.SDEProblem(dynamicfn, x₀, (0, horizonsimulation), dynamicparameters)

# Test solver
sol = SDE.solve(dynamicprob)

plottimes = range(0., horizonsimulation, 501)
startyear = 2025
plotyears = startyear .+ plottimes
yearlimits = extrema(plotyears)
yearticks = startyear:10:floor(Int, last(plotyears))
denseyticks = CairoMakie.LinearTicks(8)

ϵ = 0.1
φs = [ϵ, 0.5, 1 - ϵ]

function plottingoutput(solution, _)
    return simulationplotpath(solution, policies, terminal, climate), false
end

ensembleproblem = SDE.EnsembleProblem(dynamicprob; output_func = plottingoutput)
plotsummaries = SimulationPlotSummary[]
for φ₀ in φs
    Printf.@printf "Solving φ₀ = %.3f\n" φ₀
    ℓ₀ = log(φ₀ / (1 - φ₀))

    sol = SDE.solve(
        ensembleproblem;
        u0 = SA.SVector(ℓ₀, climate.m₀, firm.a₀),
        trajectories = 100,
        saveat = plottimes,
        save_everystep = false,
        dense = false
    )

    push!(plotsummaries, summarizesimulation(sol.u))
end

plotpath = get(ENV, "PLOTPATH", "figures")
figurepath = joinpath(plotpath, splitext(filename)[1], signallabel(signal), taxmethodlabel(taxmethod))
!ispath(figurepath) && mkpath(figurepath)

## Committed government
committedyears = startyear .+ committedtime
committedtemperatures = temperature.(getindex.(trajectory, 1), Ref(climate))
committedabatement = getindex.(trajectory, 2)
committedtaxdollars = committedtaxes ./ taxfactor

begin

    committedfig = CairoMakie.Figure(
        size = (
            3 * publicationdefault(:panelwidth),
            publicationdefault(:panelheight),
        ),
    )

    CairoMakie.Label(committedfig[0, 1], L"$\tau^{\mathrm{c}}_t$"; fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
    CairoMakie.Label(committedfig[0, 2], L"$a^{\mathrm{c}}_t$"; fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)
    CairoMakie.Label(committedfig[0, 3], L"$\zeta m^{\mathrm{c}}_t$"; fontsize = publicationdefault(:paneltitlefontsize), tellwidth = false)

    committedtaxaxis = CairoMakie.Axis(
        committedfig[1, 1];
        xlabel = "Year",
        ylabel = "Carbon tax [USD/tCO2e]",
        limits = (yearlimits, (0, nothing)),
        xticks = yearticks,
        yticks = denseyticks,
    )
    CairoMakie.lines!(
        committedtaxaxis,
        committedyears,
        committedtaxdollars;
        color = defaultpalette[:committed],
        linewidth = publicationdefault(:medianlinewidth),
    )

    committedabatementaxis = CairoMakie.Axis(
        committedfig[1, 2];
        xlabel = "Year",
        ylabel = "Abatement [GtCO2e/year]",
        limits = (yearlimits, (0, 1.05 * firm.e₀)),
        xticks = yearticks,
        yticks = denseyticks,
    )
    CairoMakie.lines!(
        committedabatementaxis,
        committedyears,
        committedabatement;
        color = defaultpalette[:committed],
        linewidth = publicationdefault(:medianlinewidth),
    )
    CairoMakie.hlines!(
        committedabatementaxis,
        [firm.e₀];
        color = defaultpalette[:committed],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )

    committedtemperatureaxis = CairoMakie.Axis(
        committedfig[1, 3];
        xlabel = "Year",
        ylabel = "Temperature [°C]",
        limits = (yearlimits, nothing),
        xticks = yearticks,
        yticks = denseyticks,
    )
    CairoMakie.lines!(
        committedtemperatureaxis,
        committedyears,
        committedtemperatures;
        color = defaultpalette[:committed],
        linewidth = publicationdefault(:medianlinewidth),
    )

    CairoMakie.linkxaxes!(
        committedtaxaxis,
        committedabatementaxis,
        committedtemperatureaxis,
    )
    savepublicationfigure(joinpath(figurepath, "committed-trajectories"), committedfig)

    println("Saved committed-government trajectories in ", figurepath)

    committedfig
end

## Noncomitted
begin
    nφ = length(φs)
    beliefcolormap = CairoMakie.resample_cmap(beliefgradient, 256)
    beliefcolor(φ) = beliefcolormap[
        clamp(round(Int, 1 + φ * (length(beliefcolormap) - 1)), 1, length(beliefcolormap))
    ]
    beliefcolors = beliefcolor.(φs)

    columns = nφ ≤ 3 ? nφ : ceil(Int, sqrt(nφ))
    rows = cld(nφ, columns)
    figuresize = (
        columns * publicationdefault(:panelwidth),
        rows * publicationdefault(:panelheight),
    )

    beliefsfigjoint = CairoMakie.Figure(size = figuresize)
    beliefvaluefigjoint = CairoMakie.Figure(size = figuresize)
    temperaturefigjoint = CairoMakie.Figure(size = figuresize)
    abatementfigjoint = CairoMakie.Figure(size = figuresize)
    taxfigjoint = CairoMakie.Figure(size = figuresize)

    beliefaxes = CairoMakie.Axis[]
    beliefvalueaxes = CairoMakie.Axis[]
    temperatureaxes = CairoMakie.Axis[]
    abatementaxes = CairoMakie.Axis[]
    taxaxes = CairoMakie.Axis[]

    CairoMakie.Label(beliefsfigjoint[0, 1:columns], L"Belief $\phi$"; fontsize = publicationdefault(:paneltitlefontsize))
    CairoMakie.Label(
        beliefvaluefigjoint[0, 1:columns],
        L"Value of beliefs $-\phi(1-\phi)\partial_{\phi} u$";
        fontsize = publicationdefault(:paneltitlefontsize),
    )
    CairoMakie.Label(temperaturefigjoint[0, 1:columns], L"Temperature $\chi m$"; fontsize = publicationdefault(:paneltitlefontsize))
    CairoMakie.Label(abatementfigjoint[0, 1:columns], L"Abatement $a$"; fontsize = publicationdefault(:paneltitlefontsize))
    CairoMakie.Label(taxfigjoint[0, 1:columns], L"Tax $\tau$"; fontsize = publicationdefault(:paneltitlefontsize))

    for (i, φ₀) in enumerate(φs)
        Printf.@printf "Plotting φ₀ = %.4f\n" φ₀
        plotsummary = plotsummaries[i]
        color = beliefcolors[i]
        row = cld(i, columns)
        column = mod1(i, columns)
        axistitle = L"$\phi_0 = %$(φ₀)$"
        panelxticks = (
            yearticks,
            [
                column == 1 || year != first(yearticks) ? string(year) : ""
                for year in yearticks
            ],
        )

        # State
        beliefaxis = CairoMakie.Axis(
            beliefsfigjoint[row, column];
            limits = (yearlimits, (0, 1)),
            xlabel = "Year",
            ylabel = L"Belief $\phi$",
            title = axistitle,
            xticks = panelxticks,
            yticks = denseyticks,
        )
        beliefvalueaxis = CairoMakie.Axis(
            beliefvaluefigjoint[row, column];
            limits = (yearlimits, nothing),
            xlabel = "Year",
            ylabel = "Value [bn USD]",
            title = axistitle,
            xticks = panelxticks,
            yticks = denseyticks,
        )
        temperatureaxis = CairoMakie.Axis(
            temperaturefigjoint[row, column];
            limits = (yearlimits, temperature.(extrema(grid.mgrid), Ref(climate))),
            xlabel = "Year",
            ylabel = "°C",
            title = axistitle,
            xticks = panelxticks,
            yticks = denseyticks,
        )
        abatementaxis = CairoMakie.Axis(
            abatementfigjoint[row, column];
            limits = (yearlimits, (0, 1.05 * firm.e₀)),
            xlabel = "Year",
            ylabel = "GtCO2 per year",
            title = axistitle,
            xticks = panelxticks,
            yticks = denseyticks,
        )

        push!(beliefaxes, beliefaxis)
        push!(beliefvalueaxes, beliefvalueaxis)
        push!(temperatureaxes, temperatureaxis)
        push!(abatementaxes, abatementaxis)

        plottrajectorysummary!(beliefaxis, plotyears, plotsummary.belief; color = color)
        plottrajectorysummary!(beliefvalueaxis, plotyears, plotsummary.beliefvalue; color = color)
        plottrajectorysummary!(temperatureaxis, plotyears, plotsummary.temperature; color = color)
        plottrajectorysummary!(abatementaxis, plotyears, plotsummary.abatement; color = color)
        CairoMakie.lines!(
            temperatureaxis,
            committedyears,
            committedtemperatures;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = publicationdefault(:committedlinewidth),
        )
        CairoMakie.lines!(
            abatementaxis,
            committedyears,
            committedabatement;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = publicationdefault(:committedlinewidth),
        )
        CairoMakie.hlines!(
            abatementaxis,
            [firm.e₀];
            color = defaultpalette[:guide],
            linestyle = :dot,
            linewidth = publicationdefault(:guidelinewidth),
        )

        # Policy
        τᶜtraj = [τᶜ(t) / taxfactor for t in plottimes]
        taxaxis = CairoMakie.Axis(
            taxfigjoint[row, column];
            limits = (yearlimits, (0, nothing)),
            xlabel = "Year",
            ylabel = "USD per tCO2",
            title = axistitle,
            xticks = panelxticks,
            yticks = denseyticks,
        )
        push!(taxaxes, taxaxis)
        plottrajectorysummary!(taxaxis, plotyears, plotsummary.tax; color = color)
        CairoMakie.lines!(
            taxaxis,
            plotyears,
            τᶜtraj;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = publicationdefault(:committedlinewidth),
        )
    end

    allaxes = (beliefaxes, beliefvalueaxes, temperatureaxes, abatementaxes, taxaxes)
    for axes in allaxes
        CairoMakie.linkyaxes!(axes)
        for axis in axes[2:end]
            CairoMakie.hideydecorations!(
                axis;
                grid = false,
                minorgrid = false,
            )
        end
    end

    savepublicationfigure(joinpath(figurepath, "beliefs"), beliefsfigjoint)
    savepublicationfigure(joinpath(figurepath, "belief-value"), beliefvaluefigjoint)
    savepublicationfigure(joinpath(figurepath, "temperature"), temperaturefigjoint)
    savepublicationfigure(joinpath(figurepath, "abatement"), abatementfigjoint)
    savepublicationfigure(joinpath(figurepath, "tax"), taxfigjoint)

    println("Saved figures in ", figurepath)
end

## Simulation with random ϕ₀
function reinitφ₀(problem, ctx)
    φ₀ = rand()
    ℓ₀ = log(φ₀ / (1 - φ₀))
    u0 = SA.SVector(ℓ₀, problem.u0[2], problem.u0[3])
    return SDE.remake(problem; u0 = u0)
end

function abatementoutput(solution, _)
    return simulationstatepath(solution, 3), false
end

ensembleuniqueprob = SDE.EnsembleProblem(
    dynamicprob;
    prob_func = reinitφ₀,
    output_func = abatementoutput,
)
sol = SDE.solve(
    ensembleuniqueprob,
    SDE.SOSRI();
    trajectories = 1_000,
    saveat = plottimes,
    save_everystep = false,
    dense = false,
)
randomplotsummary = trajectoryplotsummary(sol.u, 1)
sol = nothing
GC.gc()

begin
    abatementfig = CairoMakie.Figure()
    abatemnetaxis = CairoMakie.Axis(abatementfig[1, 1]; xlabel = "Year", ylabel = "GtCO2 per year", xticks = yearticks, yticks = denseyticks, limits = (yearlimits, (0, 1.05 * firm.e₀)))

    plottrajectorysummary!(abatemnetaxis, plotyears, randomplotsummary; color = :black)
    savepublicationfigure(joinpath(figurepath, "random-init-fig"), abatementfig)

    abatementfig
end
