using Revise

import Printf
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
import Plots
Plots.default(dpi = 180, label = false, linewidth = 2.)

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/climate.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/dynamics/state.jl")
includet("../src/dynamics/belief.jl")
includet("../src/dynamics/firm.jl")
includet("../src/dynamics/government.jl")

includet("../src/utils/arguments.jl")
includet("../src/utils/saving.jl")

includet("../src/solve/government/committed.jl")
includet("../src/solve/government/noncommitted.jl")

includet("../src/dynamics/simulation.jl")

## Load problem
## Save 
firm, government, signal, climate = initmodels()
government = Government(δ = 10.)

taxmethod = OneShotTax()
filename = solutionfilename(climate, government, firm)
solpath = joinpath("data", "solutions", filename)
if !isfile(solpath) throw("File $solpath not found.") end

solutionkey = uncommittedsolutionkey(signal, taxmethod)
solution, grid, taxmethod, trajectory, committedtaxes, committedtime =
    JLD2.jldopen(solpath, "r") do file
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

activeterminal = last(committedtime)
terminalabatement = last(trajectory)[2]
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

φs = [0.1, 0.2, 0.5, 0.75, 0.9, 1.0]
EnsemblePolicy = Vector{Vector{NTuple{3, Float64}}}
solutions = SciMLBase.EnsembleSolution[]
policyensembles = EnsemblePolicy[] 
for φ₀ in φs
    Printf.@printf "Solving φ₀ = %.1f\r" φ₀
    sol = SDE.solve(
        ensembleproblem,
        SDE.SOSRI();
        u0 = SA.SVector(φ₀, climate.m₀, firm.a₀),
        trajectories = 500,
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

function plottrajectorysummary!(figure, times, pathtimes, paths; color, scale = identity, interval = (0.025, 0.975), samplepaths = 50, plotkwargs...)
    isempty(paths) && return figure
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
        Plots.plot!(figure, pathtimes[pathindex], scaledpaths[pathindex]; c = color, alpha = 0.10, linewidth = 0.6, label = false)
    end

    observations(timeindex) = filter(isfinite, view(values, timeindex, :))
    lower = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[1]) for i in axes(values, 1)]
    median = [isempty(observations(i)) ? NaN : Statistics.median(observations(i)) for i in axes(values, 1)]
    upper = [isempty(observations(i)) ? NaN : Statistics.quantile(observations(i), interval[2]) for i in axes(values, 1)]

    Plots.plot!(figure, times, median; ribbon = (median .- lower, upper .- median), c = color, fillalpha = 0.18, linewidth = 2.5, label = false, plotkwargs...)

    return figure
end

figurepath = joinpath("figures", splitext(filename)[1], signallabel(signal), taxmethodlabel(taxmethod))
!ispath(figurepath) && mkpath(figurepath)

begin
    nφ = length(φs)
    beliefcolors = Plots.palette(:Dark2_3, nφ)
    beliefsfigures = Plots.Plot[]
    concentrationfigures = Plots.Plot[]
    abatemnetfigures = Plots.Plot[]
    taxfigures = Plots.Plot[]

    for (i, φ₀) in enumerate(φs)
        Printf.@printf "Plotting φ₀ = %.4f\n" φ₀
        dynamicsol = solutions[i]
        color = beliefcolors[i]

        # State
        belieffigure = Plots.plot(ylims = (0, 1), xlabel = "Year", title = L"$\phi_0 = %$(φ₀)$")
        concentrationfig = Plots.plot(ylims = extrema(grid.mgrid), xlabel = "Year", ylabel = "GtCO2", title = L"$\phi_0 = %$(φ₀)$")
        abatementfigure = Plots.plot(ylims = (0, firm.e₀), xlabel = "Year", ylabel = "GtCO2 per year", title = L"$\phi_0 = %$(φ₀)$")

        pathtimes = [path.t for path in dynamicsol.u]
        plottrajectorysummary!(belieffigure, plottimes, pathtimes, [getindex.(path.u, 1) for path in dynamicsol.u]; color = color)
        plottrajectorysummary!(concentrationfig, plottimes, pathtimes, [getindex.(path.u, 2) for path in dynamicsol.u]; color = color)
        plottrajectorysummary!(abatementfigure, plottimes, pathtimes, [getindex.(path.u, 3) for path in dynamicsol.u]; color = color)

        push!(beliefsfigures, belieffigure)
        push!(concentrationfigures, concentrationfig)
        push!(abatemnetfigures, abatementfigure)

        # Policy
        policyensemble = policyensembles[i]
        τᶜtraj = [τᶜ(t) / taxfactor for t in plottimes]
        taxfigure = Plots.plot(; xlabel = "Year", ylabel = "USD per tCO2", ylims = (0, Inf))
        plottrajectorysummary!(taxfigure, plottimes, pathtimes, [getindex.(path, 1) for path in policyensemble]; color, scale = τ -> τ / taxfactor)
        Plots.plot!(taxfigure, plottimes, τᶜtraj; c = color, linestyle = :dash, linewidth = 2, label = false)

        push!(taxfigures, taxfigure)
    end

    columns = ceil(Int, sqrt(nφ))
    rows = ceil(Int, nφ / columns)

    beliefsfigjoint = Plots.plot(beliefsfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Belief $\phi$", ylims = (0, 1))
    concentrationfigjoint = Plots.plot(concentrationfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Concentration $m$", ylims = extrema(grid.mgrid))
    abatemnetfigjoint = Plots.plot(abatemnetfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Abatement $a$", ylims = (0, firm.e₀))
    taxfigjoint = Plots.plot(taxfigures...; layout = (rows, columns), size = 1000 .* (√2, 1), plot_title = L"Tax $\tau$")

    Plots.savefig(beliefsfigjoint, joinpath(figurepath, "beliefs.png"))
    Plots.savefig(concentrationfigjoint, joinpath(figurepath, "concentration.png"))
    Plots.savefig(abatemnetfigjoint, joinpath(figurepath, "abatement.png"))
    Plots.savefig(taxfigjoint, joinpath(figurepath, "tax.png"))

    println("Saved figures in ", figurepath)
end
