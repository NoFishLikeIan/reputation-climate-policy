using Revise

import JLD2
import SciMLBase
import FastInterpolations as Itp

import LinearSolve
import SparseArrays
import StaticArrays as SA
import Statistics
import StochasticDiffEq as SDE
import OrdinaryDiffEq as ODE
# import StochasticDiffEqROCK as SROCK
import StochasticDiffEqImplicit as SDEImpl

import UnPack: @unpack
import LaTeXStrings: @L_str

import Printf
import CairoMakie
import Colors

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

savepublicationfigure = function (basename, figure)
    CairoMakie.save("$basename.pdf", figure; pt_per_unit = 1)
    CairoMakie.save("$basename.png", figure; px_per_unit = 2)
end

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

includet("colours.jl")

## Load problem
## Save
firm, government, signal, climate = initmodels()

taxmethod = OneShotTax()
filename = solutionfilename(climate, government, firm)
solpath = joinpath("data", "solutions", filename)
if !isfile(solpath) throw("File $solpath not found.") end

solutionkey = uncommittedsolutionkey(signal, taxmethod)
solution, grid, taxmethod, trajectory, committedtaxes, committedtime = JLD2.jldopen(solpath, "r") do file
    if !haskey(file, solutionkey)
        error("Uncommitted solution $solutionkey not found in $solpath.")
    end

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

parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, signal, climate, taxmethod)
policies = constructpolicies(solution, parameters, grid)

## Simulate path
x₀ = SA.SVector(0.5, climate.m₀, firm.a₀)
dynamicparameters = (policies, parameters, grid);
endtime = activeterminal

dynamicfn = SDE.SDEFunction{false}(dynamicdrift, dynamicnoise)
dynamicprob = SDE.SDEProblem(dynamicfn, x₀, (0, endtime), dynamicparameters)
ensembleproblem = SDE.EnsembleProblem(dynamicprob)
plottimes = range(0., endtime - 5.; length = 501)
startyear = 2025
plotyears = startyear .+ plottimes
yearlimits = extrema(plotyears)
yearticks = startyear:10:floor(Int, last(plotyears))
denseyticks = CairoMakie.LinearTicks(8)

ϵ = 0.025
φs = [ϵ, 0.5, 1 - ϵ]
EnsemblePolicy = Vector{Vector{NTuple{3, Float64}}}
EnsembleBeliefValue = Vector{Vector{Float64}}
solutions = SciMLBase.EnsembleSolution[]
policyensembles = EnsemblePolicy[]
beliefvalueensembles = EnsembleBeliefValue[]
for φ₀ in φs
    Printf.@printf "Solving φ₀ = %.4f\r" φ₀
    sol = SDE.solve(
        ensembleproblem,
        SDE.SOSRI();
        u0 = SA.SVector(φ₀, climate.m₀, firm.a₀),
        trajectories = 10_000,
        saveat = plottimes,
    )

    policyensemble = Vector{NTuple{3, Float64}}[]
    beliefvalueensemble = Vector{Float64}[]
    for soli in sol.u
        policytraj = [ policy(t, u, policies, parameters, grid) for (t, u) in zip(soli.t, soli.u) ]
        beliefvaluetraj = [
            interpolatebeliefvalue(t, u, policies, parameters, grid)
            for (t, u) in zip(soli.t, soli.u)
        ]
        push!(policyensemble, policytraj)
        push!(beliefvalueensemble, beliefvaluetraj)
    end

    push!(solutions, sol)
    push!(policyensembles, policyensemble)
    push!(beliefvalueensembles, beliefvalueensemble)
end

## Plot
samplepathlinewidth = 1.0
medianlinewidth = 3.5
committedlinewidth = 3.0
guidelinewidth = 2.0
samplepathopacity = 0.14
intervalopacity = 0.22
paneltitlefontsize = 20
annotationfontsize = 13
panelwidth = 300
panelheight = 320

function trajectoryvalue(t, pathtimes, pathvalues)
    isempty(pathtimes) && return NaN
    (t < first(pathtimes) || t > last(pathtimes)) && return NaN

    rightindex = searchsortedfirst(pathtimes, t)
    if rightindex ≤ length(pathtimes) && pathtimes[rightindex] == t
        return pathvalues[rightindex]
    end

    (rightindex == 1 || rightindex > length(pathtimes)) && return NaN
    leftindex = rightindex - 1
    weight = (t - pathtimes[leftindex]) / (pathtimes[rightindex] - pathtimes[leftindex])

    return (1 - weight) * pathvalues[leftindex] + weight * pathvalues[rightindex]
end

function plottrajectorysummary!(axis, times, pathtimes, paths; color, scale = identity, interval = (0.025, 0.975), samplepaths = 50, plotkwargs...)
    isempty(paths) && return axis
    length(pathtimes) == length(paths) || throw(DimensionMismatch("Each path needs its own time vector."))

    scaledpaths = [scale.(path) for path in paths]
    for (pathindex, (path_times, path_values)) in enumerate(zip(pathtimes, scaledpaths))
        length(path_times) == length(path_values) || throw(DimensionMismatch("Path $pathindex has different numbers of times and values."))
        issorted(path_times) || throw(ArgumentError("Times for path $pathindex are not sorted."))
        isempty(path_values) && throw(ArgumentError("Path $pathindex is empty."))
    end

    npaths = length(scaledpaths)
    values = Matrix{Float64}(undef, length(times), npaths)
    for pathindex in eachindex(scaledpaths)
        values[:, pathindex] .= trajectoryvalue.(times, Ref(pathtimes[pathindex]), Ref(scaledpaths[pathindex]))
    end

    # Stratify the displayed paths by their terminal outcome so that small
    # branches are less likely to disappear from the subsample.
    terminalorder = sortperm(last.(scaledpaths))
    sampleranks = unique(round.(Int, range(1, npaths; length = min(samplepaths, npaths))))
    sampleindices = terminalorder[sampleranks]

    for pathindex in sampleindices
        CairoMakie.lines!(
            axis,
            pathtimes[pathindex],
            scaledpaths[pathindex];
            color = (color, samplepathopacity),
            linewidth = samplepathlinewidth,
        )
    end

    observations(timeindex) = filter(isfinite, view(values, timeindex, :))
    lower = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[1]) for i in axes(values, 1)]
    median = [isempty(observations(i)) ? NaN : Statistics.median(observations(i)) for i in axes(values, 1)]
    upper = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[2]) for i in axes(values, 1)]

    CairoMakie.band!(axis, times, lower, upper; color = (color, intervalopacity))
    CairoMakie.lines!(axis, times, median; color = color, linewidth = medianlinewidth, plotkwargs...)

    return axis
end

figurepath = joinpath("figures", splitext(filename)[1], signallabel(signal), taxmethodlabel(taxmethod))
!ispath(figurepath) && mkpath(figurepath)

## Committed government
committedyears = startyear .+ committedtime
committedtemperatures = temperature.(getindex.(trajectory, 1), Ref(climate))
committedabatement = getindex.(trajectory, 2)
committedtaxdollars = committedtaxes ./ taxfactor

begin

    committedfig = CairoMakie.Figure(size = (3 * panelwidth, panelheight))

    CairoMakie.Label(committedfig[0, 1], L"$\tau^{\mathrm{c}}_t$"; fontsize = paneltitlefontsize, tellwidth = false)
    CairoMakie.Label(committedfig[0, 2], L"$a^{\mathrm{c}}_t$"; fontsize = paneltitlefontsize, tellwidth = false)
    CairoMakie.Label(committedfig[0, 3], L"$\zeta m^{\mathrm{c}}_t$"; fontsize = paneltitlefontsize, tellwidth = false)

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
        linewidth = medianlinewidth,
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
        linewidth = medianlinewidth,
    )
    CairoMakie.hlines!(
        committedabatementaxis,
        [firm.e₀];
        color = defaultpalette[:committed],
        linestyle = :dot,
        linewidth = guidelinewidth,
    )
    CairoMakie.text!(
        committedabatementaxis,
        last(committedyears),
        firm.e₀;
        text = "Net zero",
        align = (:right, :bottom),
        offset = (0, 4),
        color = defaultpalette[:committed],
        fontsize = annotationfontsize,
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
        linewidth = medianlinewidth,
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
    figuresize = (columns * panelwidth, rows * panelheight)

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

    CairoMakie.Label(beliefsfigjoint[0, 1:columns], L"Belief $\phi$"; fontsize = paneltitlefontsize)
    CairoMakie.Label(
        beliefvaluefigjoint[0, 1:columns],
        L"Value of beliefs $-\phi(1-\phi)\partial_{\phi} u$";
        fontsize = paneltitlefontsize,
    )
    CairoMakie.Label(temperaturefigjoint[0, 1:columns], L"Temperature $T$"; fontsize = paneltitlefontsize)
    CairoMakie.Label(abatementfigjoint[0, 1:columns], L"Abatement $a$"; fontsize = paneltitlefontsize)
    CairoMakie.Label(taxfigjoint[0, 1:columns], L"Tax $\tau$"; fontsize = paneltitlefontsize)

    for (i, φ₀) in enumerate(φs)
        Printf.@printf "Plotting φ₀ = %.4f\n" φ₀
        dynamicsol = solutions[i]
        color = beliefcolors[i]
        row = cld(i, columns)
        column = mod1(i, columns)
        axistitle = L"$\phi_0 = %$(φ₀)$"

        # State
        beliefaxis = CairoMakie.Axis(
            beliefsfigjoint[row, column];
            limits = (yearlimits, (0, 1)),
            xlabel = "Year",
            title = axistitle,
            xticks = yearticks,
            yticks = denseyticks,
        )
        beliefvalueaxis = CairoMakie.Axis(
            beliefvaluefigjoint[row, column];
            limits = (yearlimits, nothing),
            xlabel = "Year",
            ylabel = "Value [bn USD]",
            title = axistitle,
            xticks = yearticks,
            yticks = denseyticks,
        )
        temperatureaxis = CairoMakie.Axis(
            temperaturefigjoint[row, column];
            limits = (yearlimits, temperature.(extrema(grid.mgrid), Ref(climate))),
            xlabel = "Year",
            ylabel = "°C",
            title = axistitle,
            xticks = yearticks,
            yticks = denseyticks,
        )
        abatementaxis = CairoMakie.Axis(
            abatementfigjoint[row, column];
            limits = (yearlimits, (0, 1.05 * firm.e₀)),
            xlabel = "Year",
            ylabel = "GtCO2 per year",
            title = axistitle,
            xticks = yearticks,
            yticks = denseyticks,
        )

        push!(beliefaxes, beliefaxis)
        push!(beliefvalueaxes, beliefvalueaxis)
        push!(temperatureaxes, temperatureaxis)
        push!(abatementaxes, abatementaxis)

        pathyears = [startyear .+ path.t for path in dynamicsol.u]
        plottrajectorysummary!(beliefaxis, plotyears, pathyears, [getindex.(path.u, 1) for path in dynamicsol.u]; color = color)
        plottrajectorysummary!(
            beliefvalueaxis,
            plotyears,
            pathyears,
            beliefvalueensembles[i];
            color = color,
            scale = value -> 1_000 * value,
        )
        plottrajectorysummary!(
            temperatureaxis,
            plotyears,
            pathyears,
            [getindex.(path.u, 2) for path in dynamicsol.u];
            color = color,
            scale = m -> temperature(m, climate),
        )
        plottrajectorysummary!(abatementaxis, plotyears, pathyears, [getindex.(path.u, 3) for path in dynamicsol.u]; color = color)
        CairoMakie.lines!(
            temperatureaxis,
            committedyears,
            committedtemperatures;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = committedlinewidth,
        )
        CairoMakie.lines!(
            abatementaxis,
            committedyears,
            committedabatement;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = committedlinewidth,
        )
        CairoMakie.hlines!(
            abatementaxis,
            [firm.e₀];
            color = defaultpalette[:guide],
            linestyle = :dot,
            linewidth = guidelinewidth,
        )

        # Policy
        policyensemble = policyensembles[i]
        τᶜtraj = [τᶜ(t) / taxfactor for t in plottimes]
        taxaxis = CairoMakie.Axis(
            taxfigjoint[row, column];
            limits = (yearlimits, (0, nothing)),
            xlabel = "Year",
            ylabel = "USD per tCO2",
            title = axistitle,
            xticks = yearticks,
            yticks = denseyticks,
        )
        push!(taxaxes, taxaxis)
        plottrajectorysummary!(taxaxis, plotyears, pathyears, [getindex.(path, 1) for path in policyensemble]; color, scale = τ -> τ / taxfactor)
        CairoMakie.lines!(
            taxaxis,
            plotyears,
            τᶜtraj;
            color = defaultpalette[:committed],
            linestyle = :dash,
            linewidth = committedlinewidth,
        )
    end

    CairoMakie.linkyaxes!(beliefaxes)
    CairoMakie.linkyaxes!(beliefvalueaxes)
    CairoMakie.linkyaxes!(temperatureaxes)
    CairoMakie.linkyaxes!(abatementaxes)
    CairoMakie.linkyaxes!(taxaxes)

    savepublicationfigure(joinpath(figurepath, "beliefs"), beliefsfigjoint)
    savepublicationfigure(joinpath(figurepath, "belief-value"), beliefvaluefigjoint)
    savepublicationfigure(joinpath(figurepath, "temperature"), temperaturefigjoint)
    savepublicationfigure(joinpath(figurepath, "abatement"), abatementfigjoint)
    savepublicationfigure(joinpath(figurepath, "tax"), taxfigjoint)

    println("Saved figures in ", figurepath)
end
