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

parameters = solution.prob.p
policies = constructpolicies(solution, parameters, grid)
## Value of reputation
t = 5.
q, W = noncommittedvalues(solution(t), parameters)

beliefvalue = similar(W)
@inbounds for index in CartesianIndices(grid)
    i, j, k = index.I
    φᵢ = grid.φgrid[i]
    beliefvalue[index] = -backwardφderivative(W, i, j, k, grid) * φᵢ * (1 - φᵢ)
end

## Plot value of reputation at the initial abatement level
figurepath = joinpath(
    "figures",
    splitext(filename)[1],
    signallabel(signal),
    taxmethodlabel(taxmethod),
)
!ispath(figurepath) && mkpath(figurepath)

abatementindex = argmin(abs.(grid.agrid .- firm.a₀))
temperatures = temperature.(grid.mgrid, Ref(climate))
beliefvaluebillions = 1_000 .* beliefvalue[:, :, abatementindex]
denseyticks = CairoMakie.LinearTicks(8)

begin
    reputationfig = CairoMakie.Figure(size = (1_200, 550))
    CairoMakie.Label(
        reputationfig[0, 1:3],
        L"Value of reputation $-\partial_\phi W$";
        fontsize = 24,
    )

    heatmapaxis = CairoMakie.Axis(
        reputationfig[1, 1];
        xlabel = L"Belief $\phi$",
        ylabel = "Temperature [°C]",
        title = Printf.@sprintf(
            "Initial abatement: %.1f GtCO2e/year",
            grid.agrid[abatementindex],
        ),
        xgridvisible = false,
        ygridvisible = false,
        yticks = denseyticks,
    )
    heatmapplot = CairoMakie.heatmap!(
        heatmapaxis,
        grid.φgrid,
        temperatures,
        beliefvaluebillions;
        colormap = :viridis,
    )
    CairoMakie.contour!(
        heatmapaxis,
        grid.φgrid,
        temperatures,
        beliefvaluebillions;
        color = (:white, 0.45),
        levels = 8,
        linewidth = 0.8,
    )
    CairoMakie.Colorbar(
        reputationfig[1, 2],
        heatmapplot;
        label = "Value [bn USD]",
    )

    sliceaxis = CairoMakie.Axis(
        reputationfig[1, 3];
        xlabel = L"Belief $\phi$",
        ylabel = "Value [bn USD]",
        title = "Temperature slices",
        yticks = denseyticks,
    )
    temperatureindices = unique(round.(Int, range(1, length(temperatures); length = 4)))
    slicecolors = CairoMakie.resample_cmap(:thermal, length(temperatureindices))
    for (color, temperatureindex) in zip(slicecolors, temperatureindices)
        CairoMakie.lines!(
            sliceaxis,
            grid.φgrid,
            beliefvaluebillions[:, temperatureindex];
            color = color,
            label = Printf.@sprintf("%.2f °C", temperatures[temperatureindex]),
            linewidth = 2.5,
        )
    end
    CairoMakie.axislegend(sliceaxis; position = :rt, framevisible = false)

    CairoMakie.colgap!(reputationfig.layout, 1, 12)
    CairoMakie.colgap!(reputationfig.layout, 2, 35)
    CairoMakie.save(joinpath(figurepath, "reputation-value.png"), reputationfig)

    println("Saved figure in ", figurepath)

    reputationfig
end
