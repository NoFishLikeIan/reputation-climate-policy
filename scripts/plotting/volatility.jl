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

## Configuration
σs = @. σ̂ * 2^(0, 1, 2)
volatilitylabels = ("Normal", "High", "Very high")
volatilitycolors = (defaultpalette[:committed], Colors.colorant"#D18B47", defaultpalette[:damages])

startyear = 2025
φ₀ = 0.5

# The episode calculations use common ensemble seeds and regime-specific
# feedback policies. Entry into and exit from an elevated-volatility regime are
# both unexpected. A known exit date requires resolving the continuation problem.
simulationtrajectories = 1_000
simulationseed = UInt64(11148705)

simulationpoints = 281
shocktime = 10.0
shocktimes = [5.0, shocktime, 15.0]
shockdurations = [2.0, 5.0, 10.0]
selectedshocktime = shocktime
selectedshockduration = 10.0
initialbelief = φ₀

## Load equilibria
firm, government, signal, climate = initmodels()
taxmethod = OneShotTax()

filename = solutionfilename(climate, government, firm)
solpath = joinpath("data", "solutions", filename)
isfile(solpath) || error("File $solpath not found.")

trajectory, committedtaxes, committedtime = JLD2.jldopen(solpath, "r") do file
    file["trajectory"], file["taxes"], file["time"]
end

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
terminal = committedtaxterminal(activeterminal, terminalabatement, firm, government)

activecommittedtax = Itp.linear_interp(committedtime, committedtaxes; extrap = Itp.ClampExtrap())
τᶜ = CommittedTaxPath(activecommittedtax, activeterminal, terminal, terminalabatement, firm, government)

function constructvolatilitypolicies(solution, parameters, grid)
    n = length(solution.t)
    investment = Array{Float64}(undef, size(grid)..., n)
    tax = similar(investment)

    for (i, s) in enumerate(solution.t)
        policies = noncommittedpolicies(solution(s), parameters, s)
        investment[:, :, :, i] .= policies.investment
        tax[:, :, :, i] .= policies.tax
    end

    interpolationspace = (
        grid.φgrid,
        grid.mgrid,
        grid.agrid,
        solution.t,
    )

    return (;
        tax = Itp.linear_interp(interpolationspace, tax; extrap = clampextrap),
        investment = Itp.linear_interp(
            interpolationspace,
            investment;
            extrap = clampextrap,
        ),
    )
end

function loadvolatilityequilibrium(σ)
    comparisonsignal = Signal(σ = σ)
    solutionkey = uncommittedsolutionkey(comparisonsignal, taxmethod)

    solution, grid, savedtaxmethod = JLD2.jldopen(solpath, "r") do file
        haskey(file, solutionkey) || error(
            "Uncommitted solution $solutionkey not found in $solpath.",
        )

        (
            file["$solutionkey/solution"],
            file["$solutionkey/grid"],
            file["$solutionkey/taxmethod"],
        )
    end

    typeof(savedtaxmethod) == typeof(taxmethod) || error(
        "Expected $(typeof(taxmethod)) at $solutionkey, found $(typeof(savedtaxmethod)).",
    )

    parameters = NonCommittedParameters(
        τᶜ,
        terminal,
        grid,
        firm,
        government,
        comparisonsignal,
        climate,
        savedtaxmethod,
    )
    policies = constructvolatilitypolicies(solution, parameters, grid)

    return (; σ, signal = comparisonsignal, solution, grid, parameters, policies)
end

equilibria = map(loadvolatilityequilibrium, σs)
normalequilibrium = first(equilibria)
shockequilibria = equilibria[2:end]
shocklabels = volatilitylabels[2:end]
shockcolors = volatilitycolors[2:end]

figurepath = joinpath(ENV["PLOTPATH"], splitext(filename)[1], "volatility-shocks", taxmethodlabel(taxmethod))
ispath(figurepath) || mkpath(figurepath)

function interpolatestate(array, grid, φ, m, a)
    Itp.linear_interp(
        (grid.φgrid, grid.mgrid, grid.agrid),
        array,
        (φ, m, a);
        extrap = Itp.ClampExtrap(),
    )
end

function equilibriumslice(equilibrium, t, φvalues, m, a)
    @unpack parameters, grid, solution = equilibrium

    s = noncommittedreversetime(t, parameters)
    statevalues = solution(s)
    policies = noncommittedpolicies(statevalues, parameters, s)
    values = noncommittedvalues(statevalues, parameters)
    committedtax = parameters.τᶜ(t)

    tax = Vector{Float64}(undef, length(φvalues))
    expectedtax = similar(tax)
    investment = similar(tax)
    welfare = similar(tax)
    posteriorloading = similar(tax)
    beliefvolatility = similar(tax)
    reputationloss = similar(tax)

    for i in eachindex(φvalues)
        φ = φvalues[i]
        tax[i] = interpolatestate(policies.tax, grid, φ, m, a)
        expectedtax[i] = interpolatestate(policies.expectedtax, grid, φ, m, a)
        investment[i] = interpolatestate(policies.investment, grid, φ, m, a)
        welfare[i] = interpolatestate(values.W, grid, φ, m, a)

        gap = committedtax - tax[i]
        precision = χ(tax[i], committedtax, equilibrium.signal)
        posteriorloading[i] = φ * (1 - φ) * equilibrium.signal.ϵ * gap /
            equilibrium.signal.σ^2
        beliefvolatility[i] = beliefdiffusion(precision, φ)
        reputationloss[i] = -beliefdrift(precision, φ)
    end

    return (;
        tax,
        expectedtax,
        investment,
        welfare,
        posteriorloading,
        beliefvolatility,
        reputationloss,
        taxgap = (committedtax .- tax) ./ committedtax,
        committedtax,
    )
end

function responseheatmap!(axisposition, years, beliefs, response; colorrange, xlabel, ylabel)
    axis = CairoMakie.Axis(
        axisposition;
        xlabel,
        ylabel,
        xticks = startyear:10:floor(Int, last(years)),
        yticks = 0:0.25:1,
    )
    plot = CairoMakie.heatmap!(
        axis,
        years,
        beliefs,
        response;
        colormap = :RdBu_11,
        colorrange,
    )
    if minimum(response) ≤ 0 ≤ maximum(response)
        CairoMakie.contour!(
            axis,
            years,
            beliefs,
            response;
            levels = [0.0],
            color = (:black, 0.45),
            linewidth = 1.2,
        )
    end

    return axis, plot
end

## Immediate response to an unexpected permanent increase in volatility
maptimes = range(0.0, activeterminal; length = 41)
mapyears = startyear .+ maptimes

grid = equilibria[1].grid
φindices = (firstindex(grid.φgrid) + 2):(lastindex(grid.φgrid) - 1)
mapbeliefs = collect(grid.φgrid[φindices])

committedm = Itp.linear_interp(
    committedtime,
    getindex.(trajectory, 1);
    extrap = Itp.ClampExtrap(),
)
committeda = Itp.linear_interp(
    committedtime,
    getindex.(trajectory, 2);
    extrap = Itp.ClampExtrap(),
)

function volatilityresponse(comparisonequilibrium)
    tax = Matrix{Float64}(undef, length(maptimes), length(mapbeliefs))
    expectedtax = similar(tax)
    investment = similar(tax)
    welfare = similar(tax)

    for (timeindex, t) in enumerate(maptimes)
        m = committedm(t)
        a = committeda(t)
        normal = equilibriumslice(normalequilibrium, t, mapbeliefs, m, a)
        comparison = equilibriumslice(comparisonequilibrium, t, mapbeliefs, m, a)

        tax[timeindex, :] .= 100 .* (comparison.tax .- normal.tax) ./ normal.committedtax
        expectedtax[timeindex, :] .= 100 .* (comparison.expectedtax .- normal.expectedtax) ./
            normal.committedtax
        investment[timeindex, :] .= comparison.investment .- normal.investment
        welfare[timeindex, :] .= 1_000 .* (comparison.welfare .- normal.welfare)
    end

    return (; tax, expectedtax, investment, welfare)
end

volatilityresponses = map(volatilityresponse, shockequilibria)
responsenames = (:tax, :expectedtax, :investment, :welfare)
responsetitles = (
    "(a) Implemented tax",
    "(b) Expected tax",
    "(c) Abatement investment",
    "(d) Welfare costs",
)
responsebarlabels = (
    L"$\Delta_\sigma\tau/\tau^{\mathrm{c}}$ [pp]",
    L"$\Delta_\sigma\tau^e/\tau^{\mathrm{c}}$ [pp]",
    L"$\Delta_\sigma\dot a$ [GtCO2e/year$^2$]",
    L"$\Delta_\sigma u$ [bn USD]",
)

begin
    shockmapfig = CairoMakie.Figure(size = (1_350, 700))
    mapaxes = CairoMakie.Axis[]
    for (column, responsename) in enumerate(responsenames)
        Printf.@printf "Plotting column %i and response %s" column responsename

        maximumresponse = max(
            maximum(maximum(abs, getproperty(response, responsename)) for response in volatilityresponses),
            1e-12,
        )
        colorrange = (-maximumresponse, maximumresponse)
        columnplots = Any[]
        CairoMakie.Label(
            shockmapfig[0, column],
            responsetitles[column];
            fontsize = publicationdefault(:paneltitlefontsize),
        )

        for row in eachindex(shockequilibria)
            axis, plot = responseheatmap!(
                shockmapfig[row, column],
                mapyears,
                mapbeliefs,
                getproperty(volatilityresponses[row], responsename);
                colorrange,
                xlabel = row == lastindex(shockequilibria) ? "Year" : "",
                ylabel = column == 1 ? L"Reputation $\phi$" : "",
            )
            push!(mapaxes, axis)
            push!(columnplots, plot)
            column > 1 && CairoMakie.hideydecorations!(axis; grid = false)
            row < lastindex(shockequilibria) && CairoMakie.hidexdecorations!(axis; grid = false)
        end

        CairoMakie.Colorbar(
            shockmapfig[3, column],
            first(columnplots);
            label = responsebarlabels[column],
            vertical = false,
        )
    end
    for row in eachindex(shockequilibria)
        CairoMakie.Label(
            shockmapfig[row, 0],
            "$(shocklabels[row])\nσ = $(shockequilibria[row].σ)";
            rotation = π / 2,
            fontsize = 15,
        )
    end
    CairoMakie.linkxaxes!(mapaxes...)
    CairoMakie.linkyaxes!(mapaxes...)
    CairoMakie.Label(
        shockmapfig[4, 1:4],
        L"Each response is relative to normal volatility and evaluated along $(m_t^{\mathrm{c}},a_t^{\mathrm{c}})$";
        fontsize = 12,
        color = defaultpalette[:guide],
    )

    savepublicationfigure(
        joinpath(figurepath, "volatility-shock-map"),
        shockmapfig,
    )

    shockmapfig
end

## Belief attenuation at the initial physical state
column = 1
initialobjects = map(equilibria) do equilibrium
    equilibriumslice(
        equilibrium,
        0.0,
        mapbeliefs,
        climate.m₀,
        firm.a₀,
    )
end

begin
    attenuationfig = CairoMakie.Figure(size = (930, 350))
    gapaxis = CairoMakie.Axis(
        attenuationfig[1, 1];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Tax gap $(\tau^{\mathrm{c}}-\tau)/\tau^{\mathrm{c}}$",
        title = "(a) Endogenous policy gap",
        limits = ((0, 1), nothing),
        xticks = 0:0.25:1,
        ytickformat = values -> [Printf.@sprintf("%.0f%%", 100value) for value in values],
    )
    loadingaxis = CairoMakie.Axis(
        attenuationfig[1, 2];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Posterior loading on $\mathrm{d}s$",
        title = "(b) Response to a common signal",
        limits = ((0, 1), nothing),
        xticks = 0:0.25:1,
    )
    diffusionaxis = CairoMakie.Axis(
        attenuationfig[1, 3];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Belief diffusion $\sigma_\phi$",
        title = "(c) Equilibrium belief volatility",
        limits = ((0, 1), nothing),
        xticks = 0:0.25:1,
    )

    for (equilibrium, objects, labelname, color) in zip(
        equilibria,
        initialobjects,
        volatilitylabels,
        volatilitycolors,
    )
        label = "$labelname (σ = $(equilibrium.σ))"
        CairoMakie.lines!(
            gapaxis,
            mapbeliefs,
            objects.taxgap;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
        CairoMakie.lines!(
            loadingaxis,
            mapbeliefs,
            objects.posteriorloading;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
        CairoMakie.lines!(
            diffusionaxis,
            mapbeliefs,
            objects.beliefvolatility;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
    end
    CairoMakie.Legend(
        attenuationfig[0, 1:3],
        gapaxis;
        orientation = :horizontal,
        tellwidth = false,
        title = "Signal volatility",
    )

    savepublicationfigure(
        joinpath(figurepath, "belief-attenuation"),
        attenuationfig,
    )

    attenuationfig
end

## Unexpected volatility episodes
episode(start, duration) = (; start, stop = start + duration)
baselineepisode = (; start = Inf, stop = Inf)

function episodeequilibrium(t, context)
    context.episode.start ≤ t < context.episode.stop ? context.shock : context.normal
end

function episodedrift(x, context, t)
    equilibrium = episodeequilibrium(t, context)
    φ, m, a = x
    s = noncommittedreversetime(t, equilibrium.parameters)
    tax = equilibrium.policies.tax(φ, m, a, s)
    committedtax = equilibrium.parameters.τᶜ(t)
    investment = equilibrium.policies.investment(φ, m, a, s)
    precision = χ(tax, committedtax, equilibrium.signal)

    return SA.SVector(
        beliefdrift(precision, φ),
        cumulativeemissionsdrift(a, firm),
        investment,
    )
end

function episodenoise(x, context, t)
    equilibrium = episodeequilibrium(t, context)
    φ, m, a = x
    s = noncommittedreversetime(t, equilibrium.parameters)
    tax = equilibrium.policies.tax(φ, m, a, s)
    committedtax = equilibrium.parameters.τᶜ(t)
    precision = χ(tax, committedtax, equilibrium.signal)

    return SA.SVector(beliefdiffusion(precision, φ), 0.0, 0.0)
end

simtimes = range(0.0, activeterminal; length = simulationpoints)

episodecontext(shockequilibrium, currentepisode) = (;
    normal = normalequilibrium,
    shock = shockequilibrium,
    episode = currentepisode,
)

function solveepisode(currentepisode, shockequilibrium)
    context = episodecontext(shockequilibrium, currentepisode)
    initialstate = SA.SVector(initialbelief, climate.m₀, firm.a₀)
    dynamicfunction = SDE.SDEFunction{false}(episodedrift, episodenoise)
    problem = SDE.SDEProblem(
        dynamicfunction,
        initialstate,
        (0.0, activeterminal),
        context,
    )
    ensemble = SDE.EnsembleProblem(problem)
    stops = filter(
        t -> 0 < t < activeterminal,
        [currentepisode.start, currentepisode.stop],
    )

    return SDE.solve(
        ensemble,
        SDE.SOSRI(),
        SciMLBase.EnsembleSerial();
        trajectories = simulationtrajectories,
        saveat = simtimes,
        save_everystep = false,
        dense = false,
        tstops = stops,
        seed = simulationseed,
    )
end

selectedshocktime in shocktimes || error("The selected shock time is not in shocktimes.")
selectedshockduration in shockdurations || error(
    "The selected shock duration is not in shockdurations.",
)
selectedepisode = episode(selectedshocktime, selectedshockduration)

baselinecontext = episodecontext(normalequilibrium, baselineepisode)
println("Simulating the normal-volatility benchmark")
baselinesolution = solveepisode(baselineepisode, normalequilibrium)

abatementpersistence = Array{Float64}(
    undef,
    length(shocktimes),
    length(shockdurations),
    length(shockequilibria),
)
temperaturepersistence = similar(abatementpersistence)
selectedsolutions = Vector{Any}(undef, length(shockequilibria))
selectedcontexts = [
    episodecontext(equilibrium, selectedepisode)
    for equilibrium in shockequilibria
]

for (shockindex, shockequilibrium) in enumerate(shockequilibria)
    for (timeindex, shockstart) in enumerate(shocktimes)
        for (durationindex, duration) in enumerate(shockdurations)
            shockstart + duration < activeterminal || error(
                "The episode starting at $shockstart with duration $duration exceeds the simulation horizon.",
            )
            Printf.@printf(
                "Simulating %s volatility from %d to %d\n",
                lowercase(shocklabels[shockindex]),
                round(Int, startyear + shockstart),
                round(Int, startyear + shockstart + duration),
            )
            solution = solveepisode(
                episode(shockstart, duration),
                shockequilibrium,
            )
            abatementpersistence[timeindex, durationindex, shockindex] = Statistics.median([
                last(solution.u[pathindex].u)[3] -
                    last(baselinesolution.u[pathindex].u)[3]
                for pathindex in eachindex(solution.u)
            ])
            temperaturepersistence[timeindex, durationindex, shockindex] = Statistics.median([
                temperature(last(solution.u[pathindex].u)[2], climate) -
                    temperature(last(baselinesolution.u[pathindex].u)[2], climate)
                for pathindex in eachindex(solution.u)
            ])

            if shockstart == selectedshocktime && duration == selectedshockduration
                selectedsolutions[shockindex] = solution
            else
                solution = nothing
                GC.gc()
            end
        end
    end
end

function simulationobjects(solution, context)
    ntimes = length(simtimes)
    npaths = length(solution.u)
    belief = Matrix{Float64}(undef, ntimes, npaths)
    cumulativeemissions = similar(belief)
    abatement = similar(belief)
    tax = similar(belief)
    investment = similar(belief)
    warming = similar(belief)

    for (pathindex, path) in enumerate(solution.u)
        length(path.u) == ntimes || error(
            "Path $pathindex has $(length(path.u)) saved states; expected $ntimes.",
        )
        for timeindex in eachindex(path.u)
            t = path.t[timeindex]
            state = path.u[timeindex]
            equilibrium = episodeequilibrium(t, context)
            φ, m, a = state
            s = noncommittedreversetime(t, equilibrium.parameters)
            currenttax = equilibrium.policies.tax(φ, m, a, s)
            currentinvestment = equilibrium.policies.investment(φ, m, a, s)

            belief[timeindex, pathindex] = state[1]
            cumulativeemissions[timeindex, pathindex] = state[2]
            abatement[timeindex, pathindex] = state[3]
            tax[timeindex, pathindex] = currenttax
            investment[timeindex, pathindex] = currentinvestment
            warming[timeindex, pathindex] = temperature(state[2], climate)
        end
    end

    return (; belief, cumulativeemissions, abatement, tax, investment, warming)
end

medianseries(values) = [
    Statistics.median(view(values, timeindex, :))
    for timeindex in axes(values, 1)
]

baselineobjects = simulationobjects(baselinesolution, baselinecontext)

selectedobjects = map(
    (solution, context) -> simulationobjects(solution, context),
    selectedsolutions,
    selectedcontexts,
)
simulationyears = startyear .+ simtimes

eventresponses = map(selectedobjects) do objects
    (
        tax = medianseries((objects.tax .- baselineobjects.tax) ./ taxfactor),
        belief = 100 .* medianseries(objects.belief .- baselineobjects.belief),
        abatement = medianseries(objects.abatement .- baselineobjects.abatement),
        warming = medianseries(objects.warming .- baselineobjects.warming),
    )
end

function eventaxis!(position, responses; title, ylabel)
    axis = CairoMakie.Axis(
        position;
        xlabel = "Year",
        ylabel,
        title,
        limits = (extrema(simulationyears), nothing),
        xticks = startyear:10:floor(Int, last(simulationyears)),
        yticks = CairoMakie.LinearTicks(6),
    )
    CairoMakie.vspan!(
        axis,
        startyear + selectedshocktime,
        startyear + selectedshocktime + selectedshockduration;
        color = (defaultpalette[:guide], 0.16),
    )
    CairoMakie.hlines!(
        axis,
        [0.0];
        color = defaultpalette[:guide],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )
    for (response, label, color) in zip(responses, shocklabels, shockcolors)
        CairoMakie.lines!(
            axis,
            simulationyears,
            response;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
    end

    return axis
end

begin
    eventfig = CairoMakie.Figure(size = (900, 700))
    eventtaxaxis = eventaxis!(
        eventfig[1, 1],
        getproperty.(eventresponses, :tax);
        title = "(a) Implemented tax",
        ylabel = "USD/tCO2e",
    )
    eventbeliefaxis = eventaxis!(
        eventfig[1, 2],
        getproperty.(eventresponses, :belief);
        title = "(b) Reputation",
        ylabel = "Percentage points",
    )
    eventabatementaxis = eventaxis!(
        eventfig[2, 1],
        getproperty.(eventresponses, :abatement);
        title = "(c) Installed abatement",
        ylabel = "GtCO2e/year",
    )
    eventwarmingaxis = eventaxis!(
        eventfig[2, 2],
        getproperty.(eventresponses, :warming);
        title = "(d) Temperature",
        ylabel = "°C",
    )
    CairoMakie.linkxaxes!(
        eventtaxaxis,
        eventbeliefaxis,
        eventabatementaxis,
        eventwarmingaxis,
    )
    CairoMakie.Label(
        eventfig[0, 1:2],
        Printf.@sprintf(
            "Median response to an unexpected volatility episode, %d–%d",
            round(Int, startyear + selectedshocktime),
            round(Int, startyear + selectedshocktime + selectedshockduration),
        );
        fontsize = publicationdefault(:paneltitlefontsize),
    )
    CairoMakie.Legend(
        eventfig[3, 1:2],
        eventtaxaxis;
        orientation = :horizontal,
        tellwidth = false,
        title = "Volatility during the episode",
    )
    CairoMakie.Label(
        eventfig[4, 1:2],
        "Each response is relative to normal volatility; entry and exit are unexpected.";
        fontsize = 12,
        color = defaultpalette[:guide],
    )

    savepublicationfigure(
        joinpath(figurepath, "volatility-event-study"),
        eventfig,
    )

    eventfig
end

## Persistence by shock date and duration
begin
    persistencefig = CairoMakie.Figure(size = (900, 650))
    persistenceyears = startyear .+ shocktimes
    persistencearrays = (abatementpersistence, temperaturepersistence)
    persistencetitles = (
        "(a) Abatement at the horizon",
        "(b) Temperature at the horizon",
    )
    persistencebarlabels = (
        L"$\Delta a$ [GtCO2e/year]",
        L"$\Delta(\zeta m)$ [°C]",
    )
    persistenceaxes = CairoMakie.Axis[]

    for column in eachindex(persistencearrays)
        values = persistencearrays[column]
        maximumvalue = max(maximum(abs, values), 1e-12)
        colorrange = (-maximumvalue, maximumvalue)
        columnplots = Any[]
        CairoMakie.Label(
            persistencefig[0, column],
            persistencetitles[column];
            fontsize = publicationdefault(:paneltitlefontsize),
        )

        for row in eachindex(shockequilibria)
            axis = CairoMakie.Axis(
                persistencefig[row, column];
                xlabel = row == lastindex(shockequilibria) ? "Start of episode" : "",
                ylabel = column == 1 ? "Duration [years]" : "",
                xticks = round.(Int, persistenceyears),
                yticks = shockdurations,
            )
            plot = CairoMakie.heatmap!(
                axis,
                persistenceyears,
                shockdurations,
                view(values, :, :, row);
                colormap = :RdBu_11,
                colorrange,
            )
            push!(persistenceaxes, axis)
            push!(columnplots, plot)
            column > 1 && CairoMakie.hideydecorations!(axis; grid = false)
            row < lastindex(shockequilibria) && CairoMakie.hidexdecorations!(axis; grid = false)
        end

        CairoMakie.Colorbar(
            persistencefig[3, column],
            first(columnplots);
            label = persistencebarlabels[column],
            vertical = false,
        )
    end
    for row in eachindex(shockequilibria)
        CairoMakie.Label(
            persistencefig[row, 0],
            "$(shocklabels[row])\nσ = $(shockequilibria[row].σ)";
            rotation = π / 2,
            fontsize = 15,
        )
    end
    CairoMakie.linkxaxes!(persistenceaxes...)
    CairoMakie.linkyaxes!(persistenceaxes...)

    savepublicationfigure(
        joinpath(figurepath, "volatility-persistence"),
        persistencefig,
    )

    persistencefig
end

## Welfare-flow decomposition for the selected episode
function flowcomponents(objects)
    damages = government.y₀ .* d.(objects.cumulativeemissions, Ref(climate))
    investment = investmentcost.(
        objects.abatement,
        objects.investment,
        Ref(firm),
    )
    taxation = l.(objects.tax, Ref(government))

    return (; damages, investment, taxation)
end

function discountedflow(values)
    expectedflow = [
        Statistics.mean(view(values, timeindex, :))
        for timeindex in axes(values, 1)
    ]
    discountedcost = 0.0
    for timeindex in firstindex(simtimes):(lastindex(simtimes) - 1)
        lefttime = simtimes[timeindex]
        righttime = simtimes[timeindex + 1]
        leftvalue = exp(-government.r * lefttime) * expectedflow[timeindex]
        rightvalue = exp(-government.r * righttime) * expectedflow[timeindex + 1]
        discountedcost += (righttime - lefttime) * (leftvalue + rightvalue) / 2
    end

    return government.r * discountedcost
end

function continuationcost(solution)
    t = last(simtimes)
    s = noncommittedreversetime(t, normalequilibrium.parameters)
    values = noncommittedvalues(
        normalequilibrium.solution(s),
        normalequilibrium.parameters,
    )
    continuationvalues = [
        interpolatestate(
            values.W,
            grid,
            last(path.u)[1],
            last(path.u)[2],
            last(path.u)[3],
        )
        for path in solution.u
    ]

    return exp(-government.r * t) * Statistics.mean(continuationvalues)
end

baselineflows = flowcomponents(baselineobjects)
selectedflows = map(flowcomponents, selectedobjects)
welfarecomponents = reduce(hcat, map(selectedflows, selectedsolutions) do flows, solution
    components = 1_000 .* [
        discountedflow(flows.damages) - discountedflow(baselineflows.damages),
        discountedflow(flows.investment) - discountedflow(baselineflows.investment),
        discountedflow(flows.taxation) - discountedflow(baselineflows.taxation),
        continuationcost(solution) - continuationcost(baselinesolution),
    ]
    return [components; sum(components)]
end)
welfarelabels = [
    "Climate damages",
    "Investment",
    "Taxation",
    "Continuation",
    "Total",
]

begin
    welfarefig = CairoMakie.Figure(size = (760, 440))
    welfareaxis = CairoMakie.Axis(
        welfarefig[1, 1];
        ylabel = "Change in annualised welfare costs [bn USD]",
        title = "Welfare effect of the selected volatility episode",
        xticks = (eachindex(welfarelabels), welfarelabels),
        xticklabelrotation = π / 8,
        yticks = CairoMakie.LinearTicks(7),
    )
    CairoMakie.hlines!(
        welfareaxis,
        [0.0];
        color = defaultpalette[:guide],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )
    for shockindex in eachindex(shockequilibria)
        CairoMakie.barplot!(
            welfareaxis,
            eachindex(welfarelabels),
            view(welfarecomponents, :, shockindex);
            color = shockcolors[shockindex],
            dodge = fill(shockindex, length(welfarelabels)),
            n_dodge = length(shockequilibria),
            label = shocklabels[shockindex],
        )
    end
    CairoMakie.Legend(
        welfarefig[0, 1],
        welfareaxis;
        orientation = :horizontal,
        tellwidth = false,
        title = "Volatility during the episode",
    )
    CairoMakie.Label(
        welfarefig[2, 1],
        "Positive values are welfare-cost increases; continuation is valued under normal volatility.";
        fontsize = 12,
        color = defaultpalette[:guide],
    )

    savepublicationfigure(
        joinpath(figurepath, "volatility-welfare-decomposition"),
        welfarefig,
    )

    welfarefig
end

println("Saved volatility-shock figures in ", figurepath)
