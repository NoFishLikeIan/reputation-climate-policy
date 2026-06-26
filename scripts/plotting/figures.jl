## Setup
using Revise, BenchmarkTools
using Printf

using LaTeXStrings, Plots

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

publicationdefaults!()

const SIMPATH = joinpath("data", "solutions")

## Load baseline solution
firm, government, signal, climate = initmodels()

label = solutionlabel(climate, government, firm, signal)
baselinepath = joinpath(SIMPATH, "$label.jld2")
figurepath = joinpath("figures")
mkpath(figurepath)

solution = loadsolution(baselinepath);
itps = policyinterpolants(solution);
τᶜ₀ = itps.τᶜ(m₀)
eᶜ₀ = committedemissions(itps.τᶜ, m₀, government, firm)

mplotmax = m₀ + 70firm.e₀
mplotgrid = range(solution.mgrid[1], mplotmax, 251)
φplotgrid = range(0.05, 0.95, 181)
φslices = [0.25, 0.50, 0.75, 0.90]
φpalette = beliefspalette(length(φslices));
percentageformatter = @closure x -> @sprintf "%.0f%%" 100x
φlabel(φ) = latexstring("\\phi = " * @sprintf("%.2f", φ))

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
        c = beliefsgradient,
        levels = 20,
        linewidth = 0.,
        size = (720, 500),
        right_margin = 8Plots.mm,
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
        cbar_title = L"Welfare Costs $u$ [tUSD]",
        c = beliefsgradient,
        levels = 20,
        linewidth = 0.,
        size = (720, 500),
        right_margin = 8Plots.mm,
    )

    safesavefigure(valuefig, joinpath(figurepath, "interior-costs.png"))

    valuefig
end

## Reputation loss
begin
    taxslicefig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = "Carbon tax [USD / tCO2e]",
        xlims = extrema(mplotgrid)
    )

    for (i, φ) in enumerate(φslices)
        plot!(taxslicefig, mplotgrid, m -> itps.τ((φ, m)) / taxfactor; c = φpalette[i], label = φlabel(φ))
    end

    plot!(taxslicefig, mplotgrid, m -> itps.τᶜ(m) / taxfactor; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $\tau^c$")

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
    )

    for (i, φ) in enumerate(φslices)
        egrid = [e(aᵇ(itps.τ((φ, m)), φ, itps.τᶜ(m), government, firm), firm) for m in mplotgrid]
        plot!(emissionsslicefig, mplotgrid, egrid; c = φpalette[i], label = φlabel(φ))
    end

    plot!(emissionsslicefig, mplotgrid, m -> committedemissions(itps.τᶜ, m, government, firm); c = beliefscolors[:text], linestyle = :dash, label = L"Committed $e^c$")

    safesavefigure(emissionsslicefig, joinpath(figurepath, "reputation-loss-emissions.png"))

    emissionsslicefig
end

reputationslicesfig = plot(taxslicefig, emissionsslicefig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(reputationslicesfig, joinpath(figurepath, "reputation-loss-slices.png"))

begin
    currentφgrid = range(0.05, 0.95, 301)
    currenttax = [itps.τ((φ, m₀)) / taxfactor for φ in currentφgrid]
    currentemissions = [e(aᵇ(itps.τ((φ, m₀)), φ, itps.τᶜ(m₀), government, firm), firm) for φ in currentφgrid]

    currenttaxfig = plot(
        currentφgrid,
        currenttax;
        xlabel = L"Reputation $\phi$",
        ylabel = "Carbon tax [USD / tCO2e]",
        c = beliefscolors[:red],
        xlims = extrema(currentφgrid),
        title = L"Current-emissions policy, $m=m_0$",
    )
    hline!(currenttaxfig, [τᶜ₀ / taxfactor]; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $\tau^c$")
    
    safesavefigure(currenttaxfig, joinpath(figurepath, "reputation-loss-current-tax.png"))

    currenttaxfig
end

begin
    currentemissionsfig = plot(
        currentφgrid,
        currentemissions;
        xlabel = L"Reputation $\phi$",
        ylabel = L"Emissions $e_t$ [GtCO2e / year]",
        c = beliefscolors[:green],
        xlims = extrema(currentφgrid),
        ylims = (0, Inf),
        title = L"Current-emissions outcome, $m=m_0$",
    )
    hline!(currentemissionsfig, [eᶜ₀]; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $e^c$")

    safesavefigure(currentemissionsfig, joinpath(figurepath, "reputation-loss-current-emissions.png"))

    currentemissionsfig
end

currentpolicyfig = plot(currenttaxfig, currentemissionsfig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(currentpolicyfig, joinpath(figurepath, "reputation-loss-current-policy.png"))

## Noise comparison
lownoise = Signal()
highnoise = Signal(σ = 2lownoise.σ)

begin
    
    σspecs = ((lownoise, L"Baseline $\sigma$"), (highnoise, L"$2 \times$ baseline $\sigma$"))

    σsolutions = map(σspecs) do (σsignal, σlegend)
        σfilelabel = solutionlabel(climate, government, firm, σsignal)
        σpath = joinpath(SIMPATH, "$σfilelabel.jld2")
        σsolution = loadsolution(σpath)
        (; label = σfilelabel, plotlabel = σlegend, σ = σsignal.σ, solution = σsolution, itps = policyinterpolants(σsolution))
    end

    noisefig = plot(
        xlabel = L"Reputation $\phi$",
        ylabel = "Carbon tax [USD / tCO2e]",
        xlims = extrema(currentφgrid),
        legend = :topleft,
        title = L"Noise effect at $m=m_0$",
    )

    σpalette = beliefspalette(length(σsolutions))
    for (i, spec) in enumerate(σsolutions)
        plot!(noisefig, currentφgrid, φ -> spec.itps.τ((φ, m₀)) / taxfactor; label = spec.plotlabel, c = σpalette[i])
    end

    hline!(noisefig, [τᶜ₀ / taxfactor]; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $\tau^c$")

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
        plot!(noisedifffig, mplotgrid, Δτ; c = φpalette[i], label = φlabel(φ))
    end

    hline!(noisedifffig, [0.]; c = beliefscolors[:text], linestyle = :dash, label = "No difference")
    safesavefigure(noisedifffig, joinpath(figurepath, "noise-tax-difference.png"))

    noisecombinedfig = plot(noisefig, noisedifffig; size = (1120, 430), margins = 8Plots.mm)
    safesavefigure(noisecombinedfig, joinpath(figurepath, "noise-comparison.png"))

    noisecombinedfig
end

## Simulated paths
Random.seed!(11148705)
φ₀grid = [0.01, 0.50, 0.99]
simulation = simulatepolicies(solution, government, firm, signal; φ₀grid, horizon = 80., trajectories = 10_000)

## Plot simulated paths
pathpalette = beliefspalette(length(φ₀grid))

beliefpathfig = plot(
    xlabel = "Year",
    ylabel = L"Reputation $\phi_t$",
    ylims = (0, 1),
    legend_title = L"\phi"
)

taxpathfig = plot(
    xlabel = "Year",
    ylabel = "Carbon tax [USD / tCO2e]",
    legend_title = L"\phi"
)

emissionspathfig = plot(
    xlabel = "Year",
    ylabel = L"Emissions $e_t$ [GtCO2e / year]",
    legend_title = L"\phi"
)

cumulativepathfig = plot(
    xlabel = "Year",
    ylabel = L"Cumulative emissions $m_t$ [GtCO2e]",
    legend_title = L"\phi"
)

perceivedtaxfig = plot(
    xlabel = "Year",
    ylabel = L"Perceived carbon tax $\tau_t^e$ [USD / tCO2e]",
    legend_title = L"\phi"
)

committedpath = committedtrajectory(itps.τᶜ, simulation.timesteps, government, firm)
plot!(taxpathfig, simulation.timesteps, committedpath.τ ./ taxfactor; c = beliefscolors[:text], linestyle = :dash, label = "Committed")
plot!(emissionspathfig, simulation.timesteps, committedpath.e; c = beliefscolors[:text], linestyle = :dash, label = "Committed")
plot!(cumulativepathfig, simulation.timesteps, committedpath.m; c = beliefscolors[:text], linestyle = :dash, label = "Committed")

fillalpha = .25
for (i, φ₀) in enumerate(φ₀grid)
    policies = simulation.policies[i]
    label = φlabel(φ₀)

    plotmedian!(beliefpathfig, simulation.timesteps, policies, :φ; c = pathpalette[i], label, fillalpha)
    plotmedian!(taxpathfig, simulation.timesteps, policies, :τ; scale = x -> x / taxfactor, c = pathpalette[i], label, fillalpha)
    plotmedian!(emissionspathfig, simulation.timesteps, policies, :e; c = pathpalette[i], label, fillalpha)
    plotmedian!(cumulativepathfig, simulation.timesteps, policies, :m; c = pathpalette[i], label, fillalpha)
    plotmedian!(perceivedtaxfig, simulation.timesteps, policies, :τᵉ; scale = x -> x / taxfactor, c = pathpalette[i], label, fillalpha)

end

simulationfig = plot(beliefpathfig, taxpathfig, emissionspathfig, cumulativepathfig; layout = (2, 2), size = (1120, 760), margins = 6Plots.mm)

safesavefigure(beliefpathfig, joinpath(figurepath, "simulation-belief.png"))
safesavefigure(taxpathfig, joinpath(figurepath, "simulation-tax.png"))
safesavefigure(emissionspathfig, joinpath(figurepath, "simulation-emissions.png"))
safesavefigure(cumulativepathfig, joinpath(figurepath, "simulation-cumulative-emissions.png"))
safesavefigure(perceivedtaxfig, joinpath(figurepath, "simulation-perceived-tax.png"))
safesavefigure(simulationfig, joinpath(figurepath, "simulation-paths.png"))

@printf "\nSaved figures to %s\n" figurepath

simulationfig
