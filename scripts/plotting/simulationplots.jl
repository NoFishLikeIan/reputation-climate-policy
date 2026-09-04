struct SimulationPlotPath{T <: AbstractFloat}
    values::Matrix{T}
end

Base.eltype(::Type{SimulationPlotPath{T}}) where {T} = T
Base.eltype(::SimulationPlotPath{T}) where {T} = T

struct TrajectoryPlotSummary{T <: AbstractFloat}
    lower::Vector{T}
    median::Vector{T}
    upper::Vector{T}
    samples::Matrix{T}
end

struct SimulationPlotSummary{T <: AbstractFloat}
    belief::TrajectoryPlotSummary{T}
    beliefvalue::TrajectoryPlotSummary{T}
    temperature::TrajectoryPlotSummary{T}
    abatement::TrajectoryPlotSummary{T}
    tax::TrajectoryPlotSummary{T}
end

function simulationplotpath(solution, policies, horizon, climate)
    SciMLBase.successful_retcode(solution) || error("Simulation failed with return code $(solution.retcode).")

    values = Matrix{Float32}(undef, length(solution.t), 5)
    for (timeindex, (time, state)) in enumerate(zip(solution.t, solution.u))
        φ, m, a = state
        s = noncommittedreversetime(time, horizon)

        values[timeindex, 1] = φ
        values[timeindex, 2] = 1_000 * policies.beliefvalue(φ, m, a, s)
        values[timeindex, 3] = temperature(m, climate)
        values[timeindex, 4] = a
        tax = policies.tax(φ, m, a, s)
        values[timeindex, 5] = tax / taxfactor
    end

    return SimulationPlotPath(values)
end

function simulationstatepath(solution, stateindex)
    SciMLBase.successful_retcode(solution) || error("Simulation failed with return code $(solution.retcode).")

    values = Matrix{Float32}(undef, length(solution.t), 1)
    for (timeindex, state) in enumerate(solution.u)
        values[timeindex, 1] = state[stateindex]
    end

    return SimulationPlotPath(values)
end

function trajectoryplotsummary(paths, column; interval = (0.025, 0.975), samplepaths = 50)
    isempty(paths) && throw(ArgumentError("At least one simulation path is required."))
    0 ≤ interval[1] ≤ interval[2] ≤ 1 || throw(ArgumentError("The interval must contain valid quantiles."))
    samplepaths > 0 || throw(ArgumentError("The number of sample paths must be positive."))

    ntimes = size(first(paths).values, 1)
    npaths = length(paths)
    valuetype = eltype(first(paths))
    values = Matrix{valuetype}(undef, ntimes, npaths)

    for (pathindex, path) in enumerate(paths)
        size(path.values, 1) == ntimes || throw(DimensionMismatch("Simulation paths have different lengths."))
        checkbounds(path.values, :, column)
        values[:, pathindex] .= view(path.values, :, column)
    end

    terminalorder = sortperm(view(values, ntimes, :))
    sampleranks = unique(round.(Int, range(1, npaths; length = min(samplepaths, npaths))))
    samples = values[:, terminalorder[sampleranks]]

    lower = Vector{valuetype}(undef, ntimes)
    median = similar(lower)
    upper = similar(lower)
    for timeindex in axes(values, 1)
        observations = filter(isfinite, view(values, timeindex, :))
        lower[timeindex] = isempty(observations) ? NaN : Statistics.quantile(observations, interval[1])
        median[timeindex] = isempty(observations) ? NaN : Statistics.median(observations)
        upper[timeindex] = isempty(observations) ? NaN : Statistics.quantile(observations, interval[2])
    end

    return TrajectoryPlotSummary(lower, median, upper, samples)
end

function summarizesimulation(paths; interval = (0.025, 0.975), samplepaths = 50)
    summaries = ntuple(
        column -> trajectoryplotsummary(paths, column; interval, samplepaths),
        5,
    )
    return SimulationPlotSummary(summaries...)
end

function plottrajectorysummary!(axis, times, summary::TrajectoryPlotSummary; color, plotkwargs...)
    length(times) == length(summary.median) || throw(DimensionMismatch("Plot times and trajectory summaries have different lengths."))

    for path in eachcol(summary.samples)
        CairoMakie.lines!(
            axis,
            times,
            path;
            color = (color, publicationdefault(:samplepathopacity)),
            linewidth = publicationdefault(:samplepathlinewidth),
        )
    end

    CairoMakie.band!(
        axis,
        times,
        summary.lower,
        summary.upper;
        color = (color, publicationdefault(:intervalopacity)),
    )
    CairoMakie.lines!(
        axis,
        times,
        summary.median;
        color = color,
        linewidth = publicationdefault(:medianlinewidth),
        plotkwargs...,
    )

    return axis
end
