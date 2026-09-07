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

CairoMakie.set_theme!(publicationtheme)

## Configuration
volatilitymultipliers = 2.0 .^ (-1:2)
volatilitylabels = ("Low", "Normal", "High", "Very high")
volatilitycolors = (defaultpalette[:abatement], defaultpalette[:committed], Colors.colorant"#D18B47", defaultpalette[:damages])

φ₀ = 0.5
beliefxticks = 0:0.2:1

simulationtrajectories = 1_000
simulationseed = UInt64(11148705)

simulationpoints = 281
episodestart = 10.0
episodestarttimes = [5.0, episodestart, 15.0]
episodedurations = [2.0, 5.0, 10.0]
selectedepisodestart = episodestart
selectedepisodeduration = 10.0

## Load equilibria
firm, government, signal, climate = initmodels()
taxmethod = OneShotTax()
σs = signal.σ .* volatilitymultipliers

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
horizonsimulation = 1.2terminal

function loadvolatilityequilibrium(σ)
    comparisonsignal = Signal(; ϵ = signal.ϵ, σ)
    solutionkey = uncommittedsolutionkey(comparisonsignal, taxmethod)
    solution, grid, savedtaxmethod = JLD2.jldopen(solpath, "r") do file
        haskey(file, solutionkey) || error(
            "Uncommitted solution $solutionkey not found in $solpath.",
        )
        file["$solutionkey/solution"], file["$solutionkey/grid"], file["$solutionkey/taxmethod"]
    end
    parameters = NonCommittedParameters(τᶜ, terminal, grid, firm, government, comparisonsignal, climate, savedtaxmethod)
    policies = constructpolicies(solution, parameters, grid)
    models = (firm, government, comparisonsignal, climate)
    dynamicparameters = (policies, τᶜ, terminal, models)

    return (; σ, signal = comparisonsignal, solution, grid, parameters, policies, dynamicparameters)
end

equilibria = map(loadvolatilityequilibrium, σs)
normalindex = findfirst(==(1.0), volatilitymultipliers)
episodeindices = findall(!=(1.0), volatilitymultipliers)

normalequilibrium = equilibria[normalindex]
episodeequilibria = equilibria[episodeindices]
episodelabels = map(index -> volatilitylabels[index], episodeindices)
episodecolors = map(index -> volatilitycolors[index], episodeindices)
grid = normalequilibrium.grid
figurepath = joinpath(
    plotpath,
    splitext(filename)[1],
    "volatility-shocks",
    taxmethodlabel(taxmethod),
)
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
    policies = noncommittedpolicies(solution(s), parameters, s)
    committedtax = parameters.τᶜ(t)
    tax = [interpolatestate(policies.tax, grid, φ, m, a) for φ in φvalues]
    taxgap = committedtax .- tax
    precision = χ.(tax, committedtax, Ref(equilibrium.signal))
    posteriorloading = @. φvalues * (1 - φvalues) * equilibrium.signal.ϵ * taxgap / equilibrium.signal.σ^2
    beliefvolatility = beliefdiffusion.(precision, φvalues)

    return (; taxgap = taxgap ./ committedtax, posteriorloading, beliefvolatility)
end

## Belief attenuation at the initial physical state
φindices = (firstindex(grid.φgrid) + 2):(lastindex(grid.φgrid) - 1)
attenuationbeliefs = collect(grid.φgrid[φindices])

initialobjects = map(equilibria) do equilibrium
    equilibriumslice(
        equilibrium,
        0.0,
        attenuationbeliefs,
        climate.m₀,
        firm.a₀,
    )
end

begin
    attenuationfig = CairoMakie.Figure(
        size = (
            3 * publicationdefault(:panelwidth),
            publicationdefault(:panelheight) + 80,
        ),
    )
    gapaxis = CairoMakie.Axis(
        attenuationfig[1, 1];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Tax gap $(\tau^{\mathrm{c}}-\tau)/\tau^{\mathrm{c}}$",
        title = "(a) Endogenous policy gap",
        limits = ((0, 1), nothing),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
        ytickformat = percenttickformat,
    )
    loadingaxis = CairoMakie.Axis(
        attenuationfig[1, 2];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Posterior loading on $\mathrm{d}s$",
        title = "(b) Response to a common signal",
        limits = ((0, 1), nothing),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )
    diffusionaxis = CairoMakie.Axis(
        attenuationfig[1, 3];
        xlabel = L"Reputation $\phi$",
        ylabel = L"Belief diffusion $\sigma_\phi$",
        title = "(c) Equilibrium belief volatility",
        limits = ((0, 1), nothing),
        xticks = beliefxticks,
        xtickformat = percenttickformat,
        yticks = denseyticks,
    )

    for (equilibrium, objects, labelname, color) in zip(
        equilibria,
        initialobjects,
        volatilitylabels,
        volatilitycolors,
    )
        label = Printf.@sprintf("%s (σ = %.2f)", labelname, equilibrium.σ)
        CairoMakie.lines!(
            gapaxis,
            attenuationbeliefs,
            objects.taxgap;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
        CairoMakie.lines!(
            loadingaxis,
            attenuationbeliefs,
            objects.posteriorloading;
            color,
            linewidth = publicationdefault(:medianlinewidth),
            label,
        )
        CairoMakie.lines!(
            diffusionaxis,
            attenuationbeliefs,
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

episodeequilibrium(t, context) = context.episode.start ≤ t < context.episode.stop ? context.alternative : context.normal
episodedrift(x, context, t) = logdynamicdrift(x, episodeequilibrium(t, context).dynamicparameters, t)
episodenoise(x, context, t) = logdynamicnoise(x, episodeequilibrium(t, context).dynamicparameters, t)

simulationtimes = range(0.0, horizonsimulation; length = simulationpoints)

episodecontext(alternativeequilibrium, currentepisode) = (;
    normal = normalequilibrium,
    alternative = alternativeequilibrium,
    episode = currentepisode,
)

function solveepisode(currentepisode, alternativeequilibrium)
    context = episodecontext(alternativeequilibrium, currentepisode)
    initiallogodds = logit(φ₀)
    initialstate = SA.SVector(initiallogodds, climate.m₀, firm.a₀)
    dynamicfunction = SDE.SDEFunction{false}(episodedrift, episodenoise)
    problem = SDE.SDEProblem(
        dynamicfunction,
        initialstate,
        (0.0, horizonsimulation),
        context,
    )
    ensemble = SDE.EnsembleProblem(problem)
    stops = filter(
        t -> 0 < t < horizonsimulation,
        [currentepisode.start, currentepisode.stop],
    )

    return SDE.solve(
        ensemble,
        SDE.SOSRI(),
        SciMLBase.EnsembleSerial();
        trajectories = simulationtrajectories,
        saveat = simulationtimes,
        save_everystep = false,
        dense = false,
        tstops = stops,
        seed = simulationseed,
    )
end

selectedepisode = episode(selectedepisodestart, selectedepisodeduration)

baselinecontext = episodecontext(normalequilibrium, baselineepisode)
println("Simulating the normal-volatility benchmark")
baselinesolution = solveepisode(baselineepisode, normalequilibrium)

abatementpersistence = Array{Float64}(
    undef,
    length(episodestarttimes),
    length(episodedurations),
    length(episodeequilibria),
)
temperaturepersistence = similar(abatementpersistence)
selectedsolutions = Vector{Any}(undef, length(episodeequilibria))
selectedcontexts = [
    episodecontext(equilibrium, selectedepisode)
    for equilibrium in episodeequilibria
]

for (episodeindex, alternativeequilibrium) in enumerate(episodeequilibria)
    for (timeindex, start) in enumerate(episodestarttimes)
        for (durationindex, duration) in enumerate(episodedurations)
            0 ≤ start < start + duration ≤ horizonsimulation || error(
                "The episode starting at $start with duration $duration exceeds the simulation horizon.",
            )
            Printf.@printf(
                "Simulating %s volatility from %d to %d\n",
                lowercase(episodelabels[episodeindex]),
                round(Int, startyear + start),
                round(Int, startyear + start + duration),
            )
            episodesolution = solveepisode(episode(start, duration), alternativeequilibrium)
            abatementpersistence[timeindex, durationindex, episodeindex] = Statistics.median([
                last(episodesolution.u[pathindex].u)[3] -
                    last(baselinesolution.u[pathindex].u)[3]
                for pathindex in eachindex(episodesolution.u)
            ])
            temperaturepersistence[timeindex, durationindex, episodeindex] = Statistics.median([
                temperature(last(episodesolution.u[pathindex].u)[2], climate) -
                    temperature(last(baselinesolution.u[pathindex].u)[2], climate)
                for pathindex in eachindex(episodesolution.u)
            ])

            if start == selectedepisodestart && duration == selectedepisodeduration
                selectedsolutions[episodeindex] = episodesolution
            else
                episodesolution = nothing
                GC.gc()
            end
        end
    end
end


function simulationobjects(simulation, context)
    ntimes = length(simulationtimes)
    npaths = length(simulation.u)
    belief = Matrix{Float64}(undef, ntimes, npaths)
    cumulativeemissions = similar(belief)
    abatement = similar(belief)
    tax = similar(belief)
    investment = similar(belief)
    warming = similar(belief)

    for (pathindex, path) in enumerate(simulation.u)
        length(path.u) == ntimes || error(
            "Path $pathindex has $(length(path.u)) saved states; expected $ntimes.",
        )
        for timeindex in eachindex(path.u)
            t = path.t[timeindex]
            state = path.u[timeindex]
            equilibrium = episodeequilibrium(t, context)
            ℓ, m, a = state
            φ = logistic(ℓ)
            s = noncommittedreversetime(t, equilibrium.parameters)
            currenttax = equilibrium.policies.tax(φ, m, a, s)
            currentinvestment = equilibrium.policies.investment(φ, m, a, s)

            belief[timeindex, pathindex] = φ
            cumulativeemissions[timeindex, pathindex] = m
            abatement[timeindex, pathindex] = a
            tax[timeindex, pathindex] = currenttax
            investment[timeindex, pathindex] = currentinvestment
            warming[timeindex, pathindex] = temperature(m, climate)
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
    (simulation, context) -> simulationobjects(simulation, context),
    selectedsolutions,
    selectedcontexts,
)
simulationyears = startyear .+ simulationtimes

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
        yticks = denseyticks,
    )
    CairoMakie.vspan!(
        axis,
        startyear + selectedepisodestart,
        startyear + selectedepisodestart + selectedepisodeduration;
        color = (defaultpalette[:guide], 0.16),
    )
    CairoMakie.hlines!(
        axis,
        [0.0];
        color = defaultpalette[:guide],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )
    for (response, label, color) in zip(responses, episodelabels, episodecolors)
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
    eventfig = CairoMakie.Figure(
        size = (
            2 * publicationdefault(:panelwidth),
            2 * publicationdefault(:panelheight) + 120,
        ),
    )
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
            round(Int, startyear + selectedepisodestart),
            round(Int, startyear + selectedepisodestart + selectedepisodeduration),
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

## Persistence by episode date and duration
begin
    episoderows = length(episodeequilibria)
    colorbarrow = episoderows + 1
    persistencefig = CairoMakie.Figure(
        size = (
            2 * publicationdefault(:panelwidth),
            episoderows * publicationdefault(:panelheight) + 90,
        ),
    )
    persistenceyears = startyear .+ episodestarttimes
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
        maximumvalue = max(maximum(abs, values), eps(Float64))
        colorrange = (-maximumvalue, maximumvalue)
        columnplots = Any[]
        CairoMakie.Label(
            persistencefig[0, column],
            persistencetitles[column];
            fontsize = publicationdefault(:paneltitlefontsize),
        )

        for row in eachindex(episodeequilibria)
            axis = CairoMakie.Axis(
                persistencefig[row, column];
                xlabel = row == lastindex(episodeequilibria) ? "Start of episode" : "",
                ylabel = column == 1 ? "Duration [years]" : "",
                xticks = round.(Int, persistenceyears),
                yticks = episodedurations,
            )
            plot = CairoMakie.heatmap!(
                axis,
                persistenceyears,
                episodedurations,
                view(values, :, :, row);
                colormap = :RdBu_11,
                colorrange,
            )
            push!(persistenceaxes, axis)
            push!(columnplots, plot)
            column > 1 && CairoMakie.hideydecorations!(axis; grid = false)
            row < lastindex(episodeequilibria) && CairoMakie.hidexdecorations!(axis; grid = false)
        end

        CairoMakie.Colorbar(
            persistencefig[colorbarrow, column],
            first(columnplots);
            label = persistencebarlabels[column],
            vertical = false,
        )
    end
    for row in eachindex(episodeequilibria)
        CairoMakie.Label(
            persistencefig[row, 0],
            Printf.@sprintf("%s\nσ = %.2f", episodelabels[row], episodeequilibria[row].σ);
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
    for timeindex in firstindex(simulationtimes):(lastindex(simulationtimes) - 1)
        lefttime = simulationtimes[timeindex]
        righttime = simulationtimes[timeindex + 1]
        leftvalue = exp(-government.r * lefttime) * expectedflow[timeindex]
        rightvalue = exp(-government.r * righttime) * expectedflow[timeindex + 1]
        discountedcost += (righttime - lefttime) * (leftvalue + rightvalue) / 2
    end

    return government.r * discountedcost
end

function continuationcost(simulation)
    t = last(simulationtimes)
    s = noncommittedreversetime(t, normalequilibrium.parameters)
    values = noncommittedvalues(
        normalequilibrium.solution(s),
        normalequilibrium.parameters,
    )
    continuationvalues = [
        interpolatestate(
            values.W,
            grid,
            logistic(last(path.u)[1]),
            last(path.u)[2],
            last(path.u)[3],
        )
        for path in simulation.u
    ]

    return exp(-government.r * t) * Statistics.mean(continuationvalues)
end

baselineflows = flowcomponents(baselineobjects)
selectedflows = map(flowcomponents, selectedobjects)
welfarecolumns = map(selectedflows, selectedsolutions) do flows, simulation
    components = 1_000 .* [
        discountedflow(flows.damages) - discountedflow(baselineflows.damages),
        discountedflow(flows.investment) - discountedflow(baselineflows.investment),
        discountedflow(flows.taxation) - discountedflow(baselineflows.taxation),
        continuationcost(simulation) - continuationcost(baselinesolution),
    ]
    return [components; sum(components)]
end
welfarecomponents = hcat(welfarecolumns...)
welfarelabels = [
    "Climate damages",
    "Investment",
    "Taxation",
    "Continuation",
    "Total",
]

begin
    welfarefig = CairoMakie.Figure(
        size = (
            2 * publicationdefault(:panelwidth),
            publicationdefault(:panelheight) + 120,
        ),
    )
    welfareaxis = CairoMakie.Axis(
        welfarefig[1, 1];
        ylabel = "Change in annualised welfare costs [bn USD]",
        title = "Welfare effect of the selected volatility episode",
        xticks = (eachindex(welfarelabels), welfarelabels),
        xticklabelrotation = π / 8,
        yticks = denseyticks,
    )
    CairoMakie.hlines!(
        welfareaxis,
        [0.0];
        color = defaultpalette[:guide],
        linestyle = :dot,
        linewidth = publicationdefault(:guidelinewidth),
    )
    for episodeindex in eachindex(episodeequilibria)
        CairoMakie.barplot!(
            welfareaxis,
            eachindex(welfarelabels),
            view(welfarecomponents, :, episodeindex);
            color = episodecolors[episodeindex],
            dodge = fill(episodeindex, length(welfarelabels)),
            n_dodge = length(episodeequilibria),
            label = episodelabels[episodeindex],
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
