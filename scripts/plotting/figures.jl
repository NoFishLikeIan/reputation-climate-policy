## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots
Plots.default(linewidth = 2.5, dpi = 250, label = false, background_color = "#FAFAFA", size = 400 .* (√2, 1))

import FastInterpolations as Itp
import JLD2
import Random
import Statistics
import UnPack: @unpack
import FastClosures: @closure

import DifferentialEquations as DE
import StochasticDiffEq as SDE

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/primitives/climate.jl")
includet("../../src/agents/government.jl")
includet("../../src/utils/arguments.jl")
includet("../../src/utils/saving.jl")

includet("../../src/solve/equilibrium.jl")
includet("../../src/solve/boundaries.jl")
includet("../../src/dynamics/state.jl")
includet("../../src/utils/analysis.jl")

includet("utils.jl")
includet("colors.jl")

const SIMPATH = joinpath("data", "solutions")

## Load baseline solution
firm, government, signal, climate = initmodels()

label = solutionlabel(climate, government, firm, signal)
baselinepath = joinpath(SIMPATH, "$label.jld2")
figurepath = joinpath("figures", label)
mkpath(figurepath)

solution = loadsolution(baselinepath);
itps = policyinterpolants(solution);
τᶜ₀ = itps.τᶜ(m₀)
eᶜ₀ = committedemissions(itps.τᶜ, m₀, government, firm)

mplotmax = m₀ + 70firm.e₀
mplotgrid = range(solution.mgrid[1], mplotmax, 251)
φplotgrid = range(0.05, 0.95, 181)
φslices = [0.25, 0.50, 0.75, 0.90]
φpalette = Plots.palette(:viridis, length(φslices));
percentageformatter = @closure x -> @sprintf "%.0f%%" 100x

## Interior objects
begin
    policyfig = contourf(
        solution.mgrid[2:(end - 1)],
        solution.φgrid[2:(end - 1)],
        solution.interiorpolicy[2:(end - 1), 2:(end - 1)] ./ taxfactor;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"Tax [USD / tCO2e]",
        title = "Optimal carbon tax",
        c = :viridis, linewidth = 1
    )

    safesavefigure(policyfig, joinpath(figurepath, "interior-policy.png"))

    policyfig
end

begin
    valuefig = contourf(
        solution.mgrid,
        solution.φgrid,
        solution.u;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"Welfare Costs [tUSD]",
        title = "Government Welfare Costs",
        c = :viridis, linewidth = 1
    )

    safesavefigure(valuefig, joinpath(figurepath, "interior-costs.png"))

    valuefig
end

## Reputation loss
begin
    taxslicefig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = "Carbon tax [USD / tCO2e]",
        xlims = extrema(mplotgrid),
        legend = :topright,
        title = "Policy after a reputation loss",
    )

    for (i, φ) in enumerate(φslices)
        plot!(taxslicefig, mplotgrid, m -> itps.τ((φ, m)) / taxfactor; c = φpalette[i], label = φ)
    end

    plot!(taxslicefig, mplotgrid, m -> itps.τᶜ(m) / taxfactor; c = :black, linestyle = :dash, label = L"Committed $\tau^c$")

    safesavefigure(taxslicefig, joinpath(figurepath, "reputation-loss-tax.png"))

    taxslicefig
end

begin
    emissionsslicefig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Emissions $e_t$ [GtCO2e / year]",
        xlims = extrema(mplotgrid),
        ylims = (0, Inf),
        legend = :bottomleft,
        title = "Emissions after a reputation loss",
    )

    for (i, φ) in enumerate(φslices)
        egrid = [e(aᵇ(itps.τ((φ, m)), φ, itps.τᶜ(m), government, firm), firm) for m in mplotgrid]
        plot!(emissionsslicefig, mplotgrid, egrid; c = φpalette[i], label = φ)
    end

    plot!(emissionsslicefig, mplotgrid, m -> committedemissions(itps.τᶜ, m, government, firm); c = :black, linestyle = :dash, label = L"Committed $e^c$")

    safesavefigure(emissionsslicefig, joinpath(figurepath, "reputation-loss-emissions.png"))

    emissionsslicefig
end

begin
    currentφgrid = range(0.05, 0.95, 301)
    currenttax = [itps.τ((φ, m₀)) / taxfactor for φ in currentφgrid]
    currentemissions = [e(aᵇ(itps.τ((φ, m₀)), φ, itps.τᶜ(m₀), government, firm), firm) for φ in currentφgrid]

    currenttaxfig = plot(
        currentφgrid,
        currenttax;
        xlabel = L"Reputation $\phi$",
        ylabel = "Carbon tax [USD / tCO2e]",
        c = :darkred,
        xlims = extrema(currentφgrid),
        title = L"Current-emissions policy, $m=m_0$",
    )
    hline!(currenttaxfig, [τᶜ₀ / taxfactor]; c = :black, linestyle = :dash, label = L"Committed $\tau^c$")
    
    safesavefigure(currenttaxfig, joinpath(figurepath, "reputation-loss-current-tax.png"))

    currenttaxfig
end

begin
    currentemissionsfig = plot(
        currentφgrid,
        currentemissions;
        xlabel = L"Reputation $\phi$",
        ylabel = L"Emissions $e_t$ [GtCO2e / year]",
        c = :darkblue,
        xlims = extrema(currentφgrid),
        ylims = (0, Inf),
        title = L"Current-emissions outcome, $m=m_0$",
    )
    hline!(currentemissionsfig, [eᶜ₀]; c = :black, linestyle = :dash, label = L"Committed $e^c$")

    safesavefigure(currentemissionsfig, joinpath(figurepath, "reputation-loss-current-emissions.png"))

    currentemissionsfig
end

## Noise comparison
lownoise = Signal()
highnoise = Signal(σ = 2lownoise.σ)

begin
    
    σspecs = (lownoise, highnoise)

    σsolutions = map(σspecs) do (σsignal)
        σlabel = solutionlabel(climate, government, firm, σsignal)
        σpath = joinpath(SIMPATH, "$σlabel.jld2")
        σsolution = loadsolution(σpath)
        (; σ = σsignal.σ, solution = σsolution, itps = policyinterpolants(σsolution))
    end

    noisefig = plot(
        xlabel = L"Reputation $\phi$",
        ylabel = "Carbon tax [USD / tCO2e]",
        xlims = extrema(currentφgrid),
        legend = :topleft,
        title = L"Noise effect at $m=m_0$",
    )

    for (i, spec) in enumerate(σsolutions)
        plot!(noisefig, currentφgrid, φ -> spec.itps.τ((φ, m₀)) / taxfactor; label = spec.σ, c = Plots.palette(:Dark2_5)[i])
    end

    hline!(noisefig, [τᶜ₀ / taxfactor]; c = :black, linestyle = :dash, label = L"Committed $\tau^c$")

    safesavefigure(noisefig, joinpath(figurepath, "noise-current-tax.png"))

    noisedifffig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = "Tax increase [USD / tCO2e]",
        xlims = extrema(mplotgrid),
        legend = :topright,
        title = "Effect of higher signal noise",
    )

    low, high = first(σsolutions), last(σsolutions)

    for (i, φ) in enumerate(φslices)
        Δτ = [high.itps.τ((φ, m)) / taxfactor - low.itps.τ((φ, m)) / taxfactor for m in mplotgrid]
        plot!(noisedifffig, mplotgrid, Δτ; c = φpalette[i], label = "φ = $φ")
    end

    hline!(noisedifffig, [0.]; c = :black, linestyle = :dash, label = "Committed")
    safesavefigure(noisedifffig, joinpath(figurepath, "noise-tax-difference.png"))

    plot(noisefig, noisedifffig; size = 600 .* (2√2, 1), margins = 10Plots.mm)
end

## Simulated paths
Random.seed!(11148705)
φ₀grid = [0.3, 0.6, 0.9]
simulation = simulatepolicies(solution, government, firm, signal; φ₀grid, horizon = 80., trajectories = 500)

pathpalette = Plots.palette(:viridis, length(φ₀grid))

beliefpathfig = plot(
    xlabel = "Year",
    ylabel = L"Reputation $\phi_t$",
    ylims = (0, 1),
    legend = :bottomleft,
    title = "Reputation dynamics",
)

taxpathfig = plot(
    xlabel = "Year",
    ylabel = "Carbon tax [USD / tCO2e]",
    ylims = (0, Inf),
    legend = :topright,
    title = "Carbon tax dynamics",
)

emissionspathfig = plot(
    xlabel = "Year",
    ylabel = L"Emissions $e_t$ [GtCO2e / year]",
    ylims = (0, Inf),
    legend = :topright,
    title = "Emissions dynamics",
)

cumulativepathfig = plot(
    xlabel = "Year",
    ylabel = L"Cumulative emissions $m_t$ [GtCO2e]",
    legend = :topleft,
    title = "Cumulative emissions",
)

for (i, φ₀) in enumerate(φ₀grid)
    policies = simulation.policies[i]
    label = "φ₀ = $φ₀"

    plotmedian!(beliefpathfig, simulation.timesteps, policies, :φ; c = pathpalette[i], label)
    plotmedian!(taxpathfig, simulation.timesteps, policies, :τ; scale = x -> x / taxfactor, c = pathpalette[i], label)
    plotmedian!(emissionspathfig, simulation.timesteps, policies, :e; c = pathpalette[i], label)
    plotmedian!(cumulativepathfig, simulation.timesteps, policies, :m; c = pathpalette[i], label)
end

committedpath = committedtrajectory(itps.τᶜ, simulation.timesteps, government, firm)
plot!(taxpathfig, simulation.timesteps, committedpath.τ ./ taxfactor; c = :black, linestyle = :dash, label = "Committed")
plot!(emissionspathfig, simulation.timesteps, committedpath.e; c = :black, linestyle = :dash, label = "Committed")
plot!(cumulativepathfig, simulation.timesteps, committedpath.m; c = :black, linestyle = :dash, label = "Committed")

simulationfig = plot(beliefpathfig, taxpathfig, emissionspathfig, cumulativepathfig; layout = (2, 2), size = 800 .* (√2, 1))
savepaperfigure(simulationfig, figurepath, "simulated-reputation-loss")

@printf "\nSaved figures to %s\n" figurepath
