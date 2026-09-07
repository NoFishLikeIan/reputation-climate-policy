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
import StochasticDiffEq as SDE
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
includet("simulationplots.jl")

CairoMakie.set_theme!(publicationtheme)

## Configuration
simulationtrajectories = 1_000
simulationpoints = 501

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

solutionkey = uncommittedsolutionkey(signal, taxmethod)
solution, grid, taxmethod = JLD2.jldopen(solpath, "r") do file
    haskey(file, solutionkey) || error(
        "Uncommitted solution $solutionkey not found in $solpath.",
    )
    file["$solutionkey/solution"], file["$solutionkey/grid"], file["$solutionkey/taxmethod"]
end

models = (firm, government, signal, climate)
parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate, taxmethod)
policies = constructpolicies(solution, parameters, grid)

## Simulate paths
x₀ = SA.SVector(0.0, climate.m₀, firm.a₀)
dynamicparameters = (policies, τᶜ, terminal, models)
dynamicfunction = SDE.SDEFunction{false}(logdynamicdrift, logdynamicnoise)
dynamicproblem = SDE.SDEProblem(dynamicfunction, x₀, (0.0, terminal), dynamicparameters)

plottimes = range(0.0, terminal; length = simulationpoints)
plotyears = startyear .+ plottimes
yearlimits = extrema(plotyears)
yearticks = startyear:10:floor(Int, last(plotyears))

function reinitφ₀(problem, _, _)
    φ₀ = clamp(rand(), eps(Float64), 1 - eps(Float64))
    ℓ₀ = logit(φ₀)
    return SDE.remake(problem; u0 = SA.SVector(ℓ₀, problem.u0[2], problem.u0[3]))
end

plottingoutput(solution, _) = (
    simulationplotpath(solution, policies, terminal, climate),
    false,
)
ensembleproblem = SDE.EnsembleProblem(
    dynamicproblem;
    prob_func = reinitφ₀,
    output_func = plottingoutput,
)
simulations = SDE.solve(
    ensembleproblem,
    SDE.SOSRI();
    trajectories = simulationtrajectories,
    saveat = plottimes,
    save_everystep = false,
    dense = false,
)
randomsummary = summarizesimulation(simulations.u)

## Plot
committedyears = startyear .+ committedtime
committedtemperatures = temperature.(getindex.(trajectory, 1), Ref(climate))
committedabatement = getindex.(trajectory, 2)
committedtaxtrajectory = [τᶜ(t) / taxfactor for t in plottimes]
figurepath = joinpath(
    plotpath,
    splitext(filename)[1],
    signallabel(signal),
    taxmethodlabel(taxmethod),
)
ispath(figurepath) || mkpath(figurepath)

begin
    randomcolor = defaultpalette[:abatement]
    randomfig = CairoMakie.Figure(
        size = (
            3 * publicationdefault(:panelwidth),
            2 * publicationdefault(:panelheight),
        ),
    )
    CairoMakie.Label(
        randomfig[0, 1:3],
        L"Dynamics with $\phi_0 \sim \mathcal{U}(0,1)$";
        fontsize = publicationdefault(:paneltitlefontsize),
    )

    beliefaxis = CairoMakie.Axis(
        randomfig[1, 1];
        xlabel = "Year",
        ylabel = L"Belief $\phi$",
        title = "(a) Reputation",
        xticks = yearticks,
        yticks = 0:0.2:1,
        ytickformat = percenttickformat,
        limits = (yearlimits, (0, 1)),
    )
    beliefvalueaxis = CairoMakie.Axis(
        randomfig[1, 2];
        xlabel = "Year",
        ylabel = "Value [bn USD]",
        title = "(b) Value of beliefs",
        xticks = yearticks,
        yticks = denseyticks,
        limits = (yearlimits, nothing),
    )
    temperatureaxis = CairoMakie.Axis(
        randomfig[1, 3];
        xlabel = "Year",
        ylabel = "°C",
        title = "(c) Temperature",
        xticks = yearticks,
        yticks = denseyticks,
        limits = (
            yearlimits,
            temperature.(extrema(grid.mgrid), Ref(climate)),
        ),
    )
    abatementaxis = CairoMakie.Axis(
        randomfig[2, 1];
        xlabel = "Year",
        ylabel = "GtCO2e per year",
        title = "(d) Abatement",
        xticks = yearticks,
        yticks = denseyticks,
        limits = (yearlimits, (0, 1.05 * firm.e₀)),
    )
    taxaxis = CairoMakie.Axis(
        randomfig[2, 2];
        xlabel = "Year",
        ylabel = "USD per tCO2e",
        title = "(e) Carbon tax",
        xticks = yearticks,
        yticks = denseyticks,
        limits = (yearlimits, (0, nothing)),
    )

    plottrajectorysummary!(
        beliefaxis,
        plotyears,
        randomsummary.belief;
        color = randomcolor,
    )
    plottrajectorysummary!(
        beliefvalueaxis,
        plotyears,
        randomsummary.beliefvalue;
        color = randomcolor,
    )
    plottrajectorysummary!(
        temperatureaxis,
        plotyears,
        randomsummary.temperature;
        color = randomcolor,
    )
    plottrajectorysummary!(
        abatementaxis,
        plotyears,
        randomsummary.abatement;
        color = randomcolor,
    )
    plottrajectorysummary!(
        taxaxis,
        plotyears,
        randomsummary.tax;
        color = randomcolor,
    )

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
    CairoMakie.lines!(
        taxaxis,
        plotyears,
        committedtaxtrajectory;
        color = defaultpalette[:committed],
        linestyle = :dash,
        linewidth = publicationdefault(:committedlinewidth),
    )

    CairoMakie.linkxaxes!(
        beliefaxis,
        beliefvalueaxis,
        temperatureaxis,
        abatementaxis,
        taxaxis,
    )

    CairoMakie.Legend(
        randomfig[2, 3],
        [
            CairoMakie.LineElement(
                color = randomcolor,
                linewidth = publicationdefault(:medianlinewidth),
            ),
            CairoMakie.PolyElement(
                color = (randomcolor, publicationdefault(:intervalopacity)),
            ),
            CairoMakie.LineElement(
                color = defaultpalette[:committed],
                linestyle = :dash,
                linewidth = publicationdefault(:committedlinewidth),
            ),
            CairoMakie.LineElement(
                color = defaultpalette[:guide],
                linestyle = :dot,
                linewidth = publicationdefault(:guidelinewidth),
            ),
        ],
        [
            "Median path",
            "95% interval",
            "Committed government",
            "Net zero",
        ];
        title = "Path",
    )

    savepublicationfigure(joinpath(figurepath, "random-init-fig"), randomfig)
    println("Saved random-initial-belief dynamics in ", figurepath)

    randomfig
end
