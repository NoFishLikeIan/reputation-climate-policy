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
figurepath = joinpath("papers", "manuscript", "june-2026", "figures")
mkpath(figurepath)

solution = loadsolution(baselinepath);
itps = policyinterpolants(solution);
τᶜ₀ = itps.τᶜ(m₀)
eᶜ₀ = committedemissions(itps.τᶜ, m₀, government, firm)

mplotmax = m₀ + 50firm.e₀
mplotgrid = range(solution.mgrid[1], mplotmax, 251)
φplotgrid = range(0.05, 0.95, 181)
φslices = [0.1, 0.5, 0.9]
φpalette = beliefspalette(length(φslices));
percentageformatter = @closure x -> @sprintf "%.0f%%" 100x
φlabel(φ) = latexstring(@sprintf("%.2f", φ))

## Interior objects
interiorm = solution.mgrid[2:(end - 1)]
interiorφ = solution.φgrid[2:(end - 1)]
interiorτitp = Itp.linear_interp((interiorφ, interiorm), solution.interiorpolicy[2:(end - 1), 2:(end - 1)] ./ taxfactor)

begin
    surface = contourf(
        interiorm,
        interiorφ,
        solution.interiorpolicy[2:(end - 1), 2:(end - 1)] ./ taxfactor;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"$\tau$ [USD / tCO2e]",
        title = "Optimal carbon tax",
        c = beliefsincreasingpalette,
        linewidth = 0.,
        size = (720, 500),
        right_margin = 8Plots.mm,
        clims = (0, netzeroτ(government, firm) / taxfactor)
    )

    safesavefigure(surface, joinpath(figurepath, "interior-policy.pdf"))

    surface
end


Δτ = [log(interiorτitp((φ, m))) - log(itps.τᶜ(m) / taxfactor) for m in interiorm, φ in interiorφ]
Δτmax = maximum(abs, Δτ)
begin
    surface = contourf(
        interiorm,
        interiorφ,
        Δτ';
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"$\log(\tau / \tau^c)$",
        c = beliefsgradient,
        linewidth = 0.,
        size = (720, 500),
        right_margin = 8Plots.mm,
        clims = (-Δτmax, Δτmax)
    )

    safesavefigure(surface, joinpath(figurepath, "interior-policy-diff.pdf"))

    surface
end

begin
    surface = contourf(
        interiorm,
        interiorφ,
        solution.interiorpolicy[2:(end - 1), 2:(end - 1)] ./ taxfactor;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"Tax [USD / tCO2e]",
        title = "Optimal carbon tax",
        c = beliefsincreasingpalette,
        linewidth = 0.,
        size = (720, 500),
        right_margin = 8Plots.mm,
        clims = (0, netzeroτ(government, firm) / taxfactor)
    )

    safesavefigure(surface, joinpath(figurepath, "interior-policy.pdf"))

    surface
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

    safesavefigure(valuefig, joinpath(figurepath, "interior-costs.pdf"))

    valuefig
end

Δφ = step(solution.φgrid)
∂ᵩu = similar(solution.u) 

for j in axes(solution.u, 2)
    ∂ᵩu[1, j] = (solution.u[2, j] - solution.u[1, j]) / Δφ
    ∂ᵩu[end, j] = (solution.u[end, j] - solution.u[end - 1, j]) / Δφ

    for i in 2:(size(solution.u, 1) - 1)
        ∂ᵩu[i, j] = (solution.u[i + 1, j] - solution.u[i - 1, j]) / (2Δφ)
    end
end

∂ᵩuitp = Itp.linear_interp((solution.φgrid, solution.mgrid), ∂ᵩu; extrap = Itp.ClampExtrap())

begin
    log∂ᵩudev = @. log(1 + government.r * ∂ᵩu / y₀)
    cmax = maximum(abs, log∂ᵩudev)

    reputationvaluefig = contourf(
        solution.mgrid,
        solution.φgrid,
        log∂ᵩudev;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Reputation $\phi$",
        cbar_title = L"$\partial_{\phi} u$ [tUSD]",
        title = "Marginal value of reputation",
        c = beliefsgradient,
        size = (720, 500), linewidth = 0.,
        clims = (-cmax, cmax)

    )

    safesavefigure(reputationvaluefig, joinpath(figurepath, "reputation-value.pdf"))

    reputationvaluefig
end

begin
    reputationvalueslicefig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Value of reputation $-\partial_{\phi} u(\phi,m)$ [tUSD]",
        xlims = extrema(mplotgrid),
        legend = :topright, ylims = (0, Inf)
    )

    for (i, φ) in enumerate(φslices)
        plot!(reputationvalueslicefig, mplotgrid, m -> -∂ᵩuitp((φ, m)); c = φpalette[i], label = φlabel(φ))
    end

    safesavefigure(reputationvalueslicefig, joinpath(figurepath, "reputation-value-slices.pdf"))

    reputationvalueslicefig
end

begin
    reputationvalueslicebelieffig = plot(
        xlabel = L"Reputation",
        ylabel = L"Value of reputation $-\partial_{\phi} u(\phi,m)$ [tUSD]",
        xlims = extrema(solution.φgrid),
        legend = :topright, ylims = (0, Inf)
    )

    mslices = [1., 1.2, 2.]
    for (i, mfactor) in enumerate(mslices)
        plot!(reputationvalueslicebelieffig, solution.φgrid, φ -> -∂ᵩuitp((φ, mfactor * m₀)); c = φpalette[i], label = latexstring(@sprintf("%.1f m_0", mfactor)))
    end

    safesavefigure(reputationvalueslicebelieffig, joinpath(figurepath, "reputation-value-belief-slices.pdf"))

    reputationvalueslicebelieffig
end

reputationvaluecombinedfig = plot(reputationvaluefig, reputationvalueslicefig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(reputationvaluecombinedfig, joinpath(figurepath, "reputation-value-comparison.pdf"))

reputationlosses = [0.01, 0.05, 0.10]
reputationlosspalette = beliefspalette(length(reputationlosses))

begin
    reputationlossvaluefig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"$u(1-\epsilon,m)-u(1,m)$ [tUSD]",
        xlims = extrema(mplotgrid),
        legend_title = L"Loss $\epsilon$",
        legend = :topright,
    )

    for (i, ε) in enumerate(reputationlosses)
        plot!(reputationlossvaluefig, mplotgrid, m -> itps.u((1 - ε, m)) - itps.u((1., m)); c = reputationlosspalette[i], label = @sprintf("%.2f", ε))
    end

    hline!(reputationlossvaluefig, [0.]; c = beliefscolors[:text], linestyle = :dash, label = false)
    safesavefigure(reputationlossvaluefig, joinpath(figurepath, "reputation-loss-value.pdf"))

    reputationlossvaluefig
end

begin
    reputationlosstaxfig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"$\tau^e(1-\epsilon,m)-\tau^c(m)$ [USD / tCO2e]",
        xlims = extrema(mplotgrid),
        legend_title = L"Loss $\epsilon$",
        legend = :topright,
    )

    for (i, ε) in enumerate(reputationlosses)
        plot!(reputationlosstaxfig, mplotgrid, m -> begin
            τelow = (1 - ε) * itps.τᶜ(m) + ε * itps.τ((1 - ε, m))
            (τelow - itps.τᶜ(m)) / taxfactor
        end; c = reputationlosspalette[i], label = @sprintf("%.2f", ε))
    end

    hline!(reputationlosstaxfig, [0.]; c = beliefscolors[:text], linestyle = :dash, label = false)
    safesavefigure(reputationlosstaxfig, joinpath(figurepath, "reputation-loss-perceived-tax-wedge.pdf"))

    reputationlosstaxfig
end

begin
    reputationlossemissionsfig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"$e(1-\epsilon,m)-e(1,m)$ [GtCO2e / year]",
        xlims = extrema(mplotgrid),
        legend_title = L"Loss $\epsilon$",
        legend = :topright,
    )

    for (i, ε) in enumerate(reputationlosses)
        plot!(reputationlossemissionsfig, mplotgrid, m -> begin
            elow = e(aᵇ(itps.τ((1 - ε, m)), 1 - ε, itps.τᶜ(m), government, firm), firm)
            ehigh = committedemissions(itps.τᶜ, m, government, firm)
            elow - ehigh
        end; c = reputationlosspalette[i], label = @sprintf("%.2f", ε))
    end

    hline!(reputationlossemissionsfig, [0.]; c = beliefscolors[:text], linestyle = :dash, label = false)
    safesavefigure(reputationlossemissionsfig, joinpath(figurepath, "reputation-loss-emissions-wedge.pdf"))

    reputationlossemissionsfig
end

reputationlosspolicyfig = plot(reputationlosstaxfig, reputationlossemissionsfig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(reputationlosspolicyfig, joinpath(figurepath, "reputation-loss-policy-wedges.pdf"))

reputationlosscombinedfig = plot(reputationlossvaluefig, reputationlosstaxfig, reputationlossemissionsfig; layout = (1, 3), size = (1320, 400), margins = 6Plots.mm)
safesavefigure(reputationlosscombinedfig, joinpath(figurepath, "reputation-loss-comparison.pdf"))

## Committed government
committeduitp = Itp.linear_interp(solution.committedmgrid, solution.uᶜ; extrap = Itp.ClampExtrap())

begin
    committedtaxfig = plot(
        mplotgrid,
        m -> itps.τᶜ(m) / taxfactor;
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = "Carbon tax [USD / tCO2e]",
        c = beliefscolors[:dark],
        xlims = extrema(mplotgrid),
        label = L"Committed $\tau^c(m)$",
    )

    safesavefigure(committedtaxfig, joinpath(figurepath, "committed-tax.pdf"))

    committedtaxfig
end

begin
    committedcostfig = plot(
        mplotgrid,
        m -> committeduitp(m);
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Welfare costs $u^c(m)$ [tUSD]",
        c = beliefscolors[:dark],
        xlims = extrema(mplotgrid),
        label = L"Committed $u^c(m)$",
    )

    safesavefigure(committedcostfig, joinpath(figurepath, "committed-costs.pdf"))

    committedcostfig
end

committedpolicycostfig = plot(committedtaxfig, committedcostfig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(committedpolicycostfig, joinpath(figurepath, "committed-policy-costs.pdf"))

begin
    committedabatementfig = plot(
        mplotgrid,
        m -> a(itps.τᶜ(m), government, firm);
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Abatement $a^c(m)$ [GtCO2e / year]",
        c = beliefscolors[:green],
        xlims = extrema(mplotgrid),
        ylims = (0, Inf),
        label = L"Committed $a^c(m)$",
    )

    safesavefigure(committedabatementfig, joinpath(figurepath, "committed-abatement.pdf"))

    committedabatementfig
end

begin
    committedemissionsfig = plot(
        mplotgrid,
        m -> committedemissions(itps.τᶜ, m, government, firm);
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = L"Emissions $e^c(m)$ [GtCO2e / year]",
        c = beliefscolors[:olive],
        xlims = extrema(mplotgrid),
        ylims = (0, Inf),
        label = L"Committed $e^c(m)$",
    )

    safesavefigure(committedemissionsfig, joinpath(figurepath, "committed-emissions.pdf"))

    committedemissionsfig
end

committedoutcomesfig = plot(committedabatementfig, committedemissionsfig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(committedoutcomesfig, joinpath(figurepath, "committed-outcomes.pdf"))

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

    safesavefigure(taxslicefig, joinpath(figurepath, "reputation-loss-tax.pdf"))

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

    safesavefigure(emissionsslicefig, joinpath(figurepath, "reputation-loss-emissions.pdf"))

    emissionsslicefig
end

reputationslicesfig = plot(taxslicefig, emissionsslicefig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(reputationslicesfig, joinpath(figurepath, "reputation-loss-slices.pdf"))

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
    
    safesavefigure(currenttaxfig, joinpath(figurepath, "reputation-loss-current-tax.pdf"))

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
        ylims = (eᶜ₀, Inf),
        title = L"Current-emissions outcome, $m=m_0$",
    )
    hline!(currentemissionsfig, [eᶜ₀]; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $e^c$")

    safesavefigure(currentemissionsfig, joinpath(figurepath, "reputation-loss-current-emissions.pdf"))

    currentemissionsfig
end

currentpolicyfig = plot(currenttaxfig, currentemissionsfig; size = (1120, 430), margins = 8Plots.mm)
safesavefigure(currentpolicyfig, joinpath(figurepath, "reputation-loss-current-policy.pdf"))

## Noise comparison
lownoise = Signal()
highnoise = Signal(σ = 2lownoise.σ)

begin
    
    σspecs = ((lownoise, L"$\sigma$"), (highnoise, L"$2 \sigma$"))

    σsolutions = map(σspecs) do (σsignal, σlegend)
        σfilelabel = solutionlabel(climate, government, firm, σsignal)
        σpath = joinpath(SIMPATH, "$σfilelabel.jld2")
        σsolution = loadsolution(σpath)
        (; label = σfilelabel, plotlabel = σlegend, σ = σsignal.σ, solution = σsolution, itps = policyinterpolants(σsolution))
    end

    noisefig = plot(
        xlabel = L"Reputation $\phi$",
        ylabel = L"Initial carbon tax $\tau_0$ [USD / tCO2e]",
        xlims = extrema(currentφgrid)
    )

    σpalette = beliefspalette(length(σsolutions))
    for (i, spec) in enumerate(σsolutions)
        plot!(noisefig, currentφgrid, φ -> spec.itps.τ((φ, m₀)) / taxfactor; label = spec.plotlabel, c = σpalette[i])
    end

    hline!(noisefig, [τᶜ₀ / taxfactor]; c = beliefscolors[:text], linestyle = :dash, label = L"Committed $\tau^c$")

    safesavefigure(noisefig, joinpath(figurepath, "noise-current-tax.pdf"))

    noisedifffig = plot(
        xlabel = L"Cumulative emissions $m$ [GtCO2e]",
        ylabel = "Tax increase [USD / tCO2e]",
        xlims = extrema(mplotgrid),
        title = "Effect of higher signal noise",
        ylims = (-Inf, 0.05)
    )

    low, high = first(σsolutions), last(σsolutions)

    for (i, φ) in enumerate(φslices)
        Δτ = [high.itps.τ((φ, m)) / taxfactor - low.itps.τ((φ, m)) / taxfactor for m in mplotgrid]
        plot!(noisedifffig, mplotgrid, Δτ; c = φpalette[i], label = φlabel(φ))
    end

    safesavefigure(noisedifffig, joinpath(figurepath, "noise-tax-difference.pdf"))

    noisecombinedfig = plot(noisefig, noisedifffig; size = (1120, 430), margins = 8Plots.mm)
    safesavefigure(noisecombinedfig, joinpath(figurepath, "noise-comparison.pdf"))

    noisecombinedfig
end

## Simulated paths
Random.seed!(11148705)
pathhorizon = 80.
pathtimesteps = range(0, pathhorizon; step = 1 / 6)
committedpath = committedtrajectory(itps.τᶜ, pathtimesteps, government, firm)

committedtaxpathfig = plot(
    pathtimesteps,
    committedpath.τ ./ taxfactor;
    xlabel = "Year",
    ylabel = "Carbon tax [USD / tCO2e]",
    c = beliefscolors[:red],
    label = L"Committed $\tau^c_t$",
    xlims = extrema(pathtimesteps),
)

committedabatementpathfig = plot(
    pathtimesteps,
    firm.e₀ .- committedpath.e;
    xlabel = "Year",
    ylabel = L"Abatement $a^c_t$ [GtCO2e / year]",
    c = beliefscolors[:green],
    label = L"Committed $a^c_t$",
    xlims = extrema(pathtimesteps),
    ylims = (0, Inf),
)

committedemissionspathfig = plot(
    pathtimesteps,
    committedpath.e;
    xlabel = "Year",
    ylabel = L"Emissions $e^c_t$ [GtCO2e / year]",
    c = beliefscolors[:olive],
    label = L"Committed $e^c_t$",
    xlims = extrema(pathtimesteps),
    ylims = (0, Inf),
)

committedcumulativepathfig = plot(
    pathtimesteps,
    committedpath.m;
    xlabel = "Year",
    ylabel = L"Cumulative emissions $m^c_t$ [GtCO2e]",
    c = beliefscolors[:dark],
    label = L"Committed $m^c_t$",
    xlims = extrema(pathtimesteps),
)

committedpathsfig = plot(
    committedtaxpathfig,
    committedabatementpathfig,
    committedemissionspathfig,
    committedcumulativepathfig;
    layout = (2, 2),
    size = (1120, 760),
    margins = 6Plots.mm,
)

safesavefigure(committedtaxpathfig, joinpath(figurepath, "committed-path-tax.pdf"))
safesavefigure(committedabatementpathfig, joinpath(figurepath, "committed-path-abatement.pdf"))
safesavefigure(committedemissionspathfig, joinpath(figurepath, "committed-path-emissions.pdf"))
safesavefigure(committedcumulativepathfig, joinpath(figurepath, "committed-path-cumulative-emissions.pdf"))
safesavefigure(committedpathsfig, joinpath(figurepath, "committed-paths.pdf"))

φ₀grid = [0.1, 0.50, 0.9]
simulation = simulatepolicies(solution, government, firm, signal; φ₀grid, horizon = pathhorizon, trajectories = 10_000)

## Plot simulated paths
pathpalette = beliefspalette(length(φ₀grid))

beliefpathfig = plot(
    xlabel = "Year", xlims = (0, pathhorizon),
    ylabel = L"Reputation $\phi_t$",
    ylims = (0, 1),
    legend = false
)

taxpathfig = plot(
    xlabel = "Year", xlims = (0, pathhorizon),
    ylabel = "Carbon tax [USD / tCO2e]",
    legend = false,
    ylims = (60, netzeroτ(government, firm) / taxfactor)
)

emissionspathfig = plot(
    xlabel = "Year", xlims = (0, pathhorizon),
    ylabel = L"Emissions $e_t$ [GtCO2e / year]",
    legend = false
)

cumulativepathfig = plot(
    xlabel = "Year", xlims = (0, pathhorizon),
    ylabel = L"Cumulative emissions $m_t$ [GtCO2e]",
    legend_title = L"\phi_0"
)

perceivedtaxfig = plot(
    xlabel = "Year", xlims = (0, pathhorizon),
    ylabel = L"Expected carbon tax $\tau_t^e$ [USD / tCO2e]",
    legend = false,
    ylims = (60, netzeroτ(government, firm) / taxfactor)
)

plot!(taxpathfig, pathtimesteps, committedpath.τ ./ taxfactor; c = beliefscolors[:text], linestyle = :dash, label = "Committed")
plot!(emissionspathfig, pathtimesteps, committedpath.e; c = beliefscolors[:text], linestyle = :dash, label = "Committed")
plot!(cumulativepathfig, pathtimesteps, committedpath.m; c = beliefscolors[:text], linestyle = :dash, label = "Committed")

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

simulationfig = plot(taxpathfig, perceivedtaxfig, beliefpathfig, cumulativepathfig; layout = (2, 2), size = (1120, 760), margins = 6Plots.mm)

safesavefigure(beliefpathfig, joinpath(figurepath, "simulation-belief.pdf"))
safesavefigure(taxpathfig, joinpath(figurepath, "simulation-tax.pdf"))
safesavefigure(emissionspathfig, joinpath(figurepath, "simulation-emissions.pdf"))
safesavefigure(cumulativepathfig, joinpath(figurepath, "simulation-cumulative-emissions.pdf"))
safesavefigure(perceivedtaxfig, joinpath(figurepath, "simulation-perceived-tax.pdf"))
safesavefigure(simulationfig, joinpath(figurepath, "simulation-paths.pdf"))

@printf "\nSaved figures to %s\n" figurepath

simulationfig
