using Revise

import JLD2
import OrdinaryDiffEq as ODE
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
plottimes = range(0., endtime; length = 501)
startyear = 2025
plotyears = startyear .+ plottimes

φs = [1e-2, 0.2, 0.5, 1 - 1e-2]
EnsemblePolicy = Vector{Vector{NTuple{3, Float64}}}
solutions = SciMLBase.EnsembleSolution[]
policyensembles = EnsemblePolicy[]
for φ₀ in φs
    Printf.@printf "Solving φ₀ = %.1f\r" φ₀
    sol = SDE.solve(
        ensembleproblem,
        SDE.SOSRI();
        u0 = SA.SVector(φ₀, climate.m₀, firm.a₀),
        trajectories = 10_000,
        saveat = plottimes,
    )

    policyensemble = Vector{NTuple{3, Float64}}[]
    for soli in sol.u
        policytraj = [ policy(t, u, policies, parameters, grid) for (t, u) in zip(soli.t, soli.u) ]
        push!(policyensemble, policytraj)
    end

    push!(solutions, sol)
    push!(policyensembles, policyensemble)
end

## Plot
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
            color = (color, 0.10),
            linewidth = 0.6,
        )
    end

    observations(timeindex) = filter(isfinite, view(values, timeindex, :))
    lower = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[1]) for i in axes(values, 1)]
    median = [isempty(observations(i)) ? NaN : Statistics.median(observations(i)) for i in axes(values, 1)]
    upper = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[2]) for i in axes(values, 1)]

    CairoMakie.band!(axis, times, lower, upper; color = (color, 0.18))
    CairoMakie.lines!(axis, times, median; color = color, linewidth = 2.5, plotkwargs...)

    return axis
end

figurepath = joinpath("figures", splitext(filename)[1], signallabel(signal), taxmethodlabel(taxmethod))
!ispath(figurepath) && mkpath(figurepath)

begin
    nφ = length(φs)
    beliefcolormap = CairoMakie.resample_cmap(beliefgradient, 256)
    beliefcolor(φ) = beliefcolormap[
        clamp(round(Int, 1 + φ * (length(beliefcolormap) - 1)), 1, length(beliefcolormap))
    ]
    beliefcolors = beliefcolor.(φs)

    columns = ceil(Int, sqrt(nφ))
    figuresize = (round(Int, 1000 * sqrt(2)), 1000)

    beliefsfigjoint = CairoMakie.Figure(size = figuresize)
    temperaturefigjoint = CairoMakie.Figure(size = figuresize)
    abatementfigjoint = CairoMakie.Figure(size = figuresize)
    taxfigjoint = CairoMakie.Figure(size = figuresize)

    CairoMakie.Label(beliefsfigjoint[0, 1:columns], L"Belief $\phi$"; fontsize = 24)
    CairoMakie.Label(temperaturefigjoint[0, 1:columns], L"Temperature $T$"; fontsize = 24)
    CairoMakie.Label(abatementfigjoint[0, 1:columns], L"Abatement $a$"; fontsize = 24)
    CairoMakie.Label(taxfigjoint[0, 1:columns], L"Tax $\tau$"; fontsize = 24)

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
            limits = (nothing, (0, 1)),
            xlabel = "Year",
            title = axistitle,
        )
        temperatureaxis = CairoMakie.Axis(
            temperaturefigjoint[row, column];
            limits = (nothing, temperature.(extrema(grid.mgrid), Ref(climate))),
            xlabel = "Year",
            ylabel = "°C",
            title = axistitle,
        )
        abatementaxis = CairoMakie.Axis(
            abatementfigjoint[row, column];
            limits = (nothing, (0, firm.e₀)),
            xlabel = "Year",
            ylabel = "GtCO2 per year",
            title = axistitle,
        )

        pathyears = [startyear .+ path.t for path in dynamicsol.u]
        plottrajectorysummary!(beliefaxis, plotyears, pathyears, [getindex.(path.u, 1) for path in dynamicsol.u]; color = color)
        plottrajectorysummary!(
            temperatureaxis,
            plotyears,
            pathyears,
            [getindex.(path.u, 2) for path in dynamicsol.u];
            color = color,
            scale = m -> temperature(m, climate),
        )
        plottrajectorysummary!(abatementaxis, plotyears, pathyears, [getindex.(path.u, 3) for path in dynamicsol.u]; color = color)

        # Policy
        policyensemble = policyensembles[i]
        τᶜtraj = [τᶜ(t) / taxfactor for t in plottimes]
        taxaxis = CairoMakie.Axis(
            taxfigjoint[row, column];
            limits = (nothing, (0, nothing)),
            xlabel = "Year",
            ylabel = "USD per tCO2",
            title = axistitle,
        )
        plottrajectorysummary!(taxaxis, plotyears, pathyears, [getindex.(path, 1) for path in policyensemble]; color, scale = τ -> τ / taxfactor)
        CairoMakie.lines!(taxaxis, plotyears, τᶜtraj; color = color, linestyle = :dash, linewidth = 2)
    end

    CairoMakie.save(joinpath(figurepath, "beliefs.png"), beliefsfigjoint)
    CairoMakie.save(joinpath(figurepath, "temperature.png"), temperaturefigjoint)
    CairoMakie.save(joinpath(figurepath, "abatement.png"), abatementfigjoint)
    CairoMakie.save(joinpath(figurepath, "tax.png"), taxfigjoint)

    println("Saved figures in ", figurepath)
end
