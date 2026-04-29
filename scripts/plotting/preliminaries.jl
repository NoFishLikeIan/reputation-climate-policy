## Setup
using Revise
using UnPack

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using Colors

using Optim
using LaTeXStrings, Printf

using Plots
default(linewidth = 4, dpi = 250, label = false, background_color = :transparent, size = 500 .* (√2, 1))
Plots.scalefontsizes(1.3)

plotpath = "papers/figures/preliminaries"
if !isdir(plotpath) mkdir(plotpath) end

includet("colors.jl")
includet("../../src/constants.jl")
includet("../../src/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/grid.jl")
includet("../../src/valuefunction.jl")
includet("../../src/boundary.jl")

const taxfactor = scctotax

firm = Firm()
government = Government()
signal = Signal(1.0, 1e-2, 41)

## Compute figure values
a0 = 0.0
τᶜ = optimize(τ -> w̄(a0, τ, firm, government, signal), 0.0, 1.0, Brent()).minimizer
φstar = φ̄(τᶜ, firm, signal)
a_bliss = blissabatement(τᶜ, firm, signal)

τspace_usd = collect(range(0.0, 250.0; length = 251))
τspace = τspace_usd .* taxfactor
aspace = collect(range(0.0, 1.25 * max(a_bliss, φstar, 1e-6); length = 250))
φspace = collect(range(0.0, 1.5 * max(φstar, 1e-6); length = 250))
ξspace = collect(range(-3.0, 3.0; length = 250))
qspace = realisedprice.(ξspace, τᶜ, Ref(signal))
zspace = collect(range(-0.25, 0.05; length = 250))
zs = [0.0, -0.025, -0.05, -0.10]

τlabel = L"Carbon tax $\tau$ (USD/tCO$_2$)"
alabel = L"Abatement stock $a$ [GtCO$_2$ / year]"
φlabel = L"Abatement investment $\phi$ [GtCO$_2$ / year]"
zlabel = L"Reputation $z$"

normalpdf(x, μ, σ) = @inline exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

## Plots
let
    signalτshares = [0.2, 0.5, 1.]
    signalτs = signalτshares .* τᶜ
    browncolor = beliefscolors[:brown]
    greencolor = beliefscolors[:green]
    signalcolors = [
        RGB(
            (1 - x) * red(browncolor) + x * red(greencolor),
            (1 - x) * green(browncolor) + x * green(greencolor),
            (1 - x) * blue(browncolor) + x * blue(greencolor),
        )
        for x in range(0.0, 1.0; length = length(signalτshares))
    ]

    signallabels = [L"%$(frac)" for frac in signalτshares]

    signalξ = 4.0
    signalqmin = minimum(realisedprice(-signalξ, τ, signal) for τ in signalτs)
    signalqmax = maximum(realisedprice(signalξ, τ, signal) for τ in signalτs)
    signalqspace = collect(range(signalqmin, signalqmax; length = 600))
    signalqspaceusd = signalqspace ./ taxfactor

    signalzτs = signalτs[1:(end - 1)]
    signalzmin = minimum(ℓ(realisedprice(-signalξ, τ, signal), τ, τᶜ, signal) for τ in signalzτs)
    signalzmax = maximum(ℓ(realisedprice(signalξ, τ, signal), τ, τᶜ, signal) for τ in signalzτs)
    signalzbound = max(abs(signalzmin), abs(signalzmax))
    signalzspace = collect(range(-signalzbound, signalzbound; length = 600))

    signaldensity(q, τ) = normalpdf(q, signal.μ * τ, signal.σ) * taxfactor
    function reputationdensity(z, τ)
        zslope = signal.μ * (τᶜ - τ) / signal.σ^2
        zcenter = signal.μ * (τ + τᶜ) / 2
        q = zcenter + z / zslope
        return normalpdf(q, signal.μ * τ, signal.σ) / abs(zslope)
    end

    densityfig = plot(
        xlabel = L"Carbon price $q$ (USD/tCO$_2$)",
        ylabel = "Density",
        legend = false,
        ylims = (0, normalpdf(0.0, 0.0, signal.σ) * taxfactor * 1.08),
        yformatter = _ -> ""
    )

    for (i, τ) in enumerate(signalτs)
        plot!(
            densityfig,
            signalqspaceusd,
            [signaldensity(q, τ) for q in signalqspace];
            c = signalcolors[i],
            label = signallabels[i],
        )
        vline!(
            densityfig,
            [signal.μ * τ / taxfactor];
            c = signalcolors[i],
            linestyle = :dot,
            label = false,
        )
    end

    zfig = plot(
        xlabel = L"Signal-implied reputation update $z^\prime-z$",
        ylabel = "Density",
        legend = :topright,
        legendtitle = L"Policy $\tau  /\tau^c$",
        yformatter = _ -> "",
        ylims = (0, Inf)
    )

    for (i, τ) in enumerate(signalzτs)
        plot!(
            zfig,
            signalzspace,
            [reputationdensity(z, τ) for z in signalzspace];
            c = signalcolors[i],
            label = signallabels[i],
        )
        vline!(
            zfig,
            [ℓ(signal.μ * τ, τ, τᶜ, signal)];
            c = signalcolors[i],
            linestyle = :dot,
            label = false,
        )
    end

    vline!(zfig, [0.0]; c = signalcolors[end], label = signallabels[end])

    fig = plot(
        densityfig,
        zfig;
        layout = (1, 2),
        size = (980, 380),
        margins = 5Plots.mm,
    )
	savefig(fig, joinpath(plotpath, "signal-distribution-reputation.png"))
    fig
end

let
    boundaryaspace = collect(range(0.0, aspace[end]; length = 250))
    signalξs, signalweights = signal.space

    expectedfirmboundary(a, τ, firmboundary) = sum(
        signalweights[i] * firmboundary(a, realisedprice(signalξs[i], τ, signal))
        for i in eachindex(signalξs)
    )

    wlower = [w̲(a, firm, government) for a in boundaryaspace]
    wupper = [w̄(a, τᶜ, firm, government, signal) for a in boundaryaspace]
    vlower = [
        expectedfirmboundary(a, τ̲(a, firm, government), (a, q) -> v̲(a, q, firm))
        for a in boundaryaspace
    ]
    vupper = [
        expectedfirmboundary(a, τ̄(τᶜ, firm, government), (a, q) -> v̄(a, q, τᶜ, firm, signal))
        for a in boundaryaspace
    ]
    boundaryymax = maximum((maximum(wlower), maximum(wupper), maximum(vlower), maximum(vupper)))
    boundaryylims = (-5., boundaryymax)
    valueylabel = "Value [trUSD]"

    wfig = plot(
        boundaryaspace,
        wlower;
        c = beliefscolors[:brown],
        xlabel = alabel,
        ylabel = valueylabel,
        label = L"\underbar{w}(a)",
        legend = :topright,
        ylims = boundaryylims,
        xlims = extrema(boundaryaspace)
    )
    plot!(
        wfig,
        boundaryaspace,
        wupper;
        c = beliefscolors[:dark],
        label = L"\bar{w}(a)",
    )

    vfig = plot(
        boundaryaspace,
        vlower;
        c = beliefscolors[:brown],
        xlabel = alabel,
        ylabel = valueylabel,
        label = L"\mathrm{E}_q[\underbar{v}(a,q)]",
        legend = :topright,
        ylims = boundaryylims,
        xlims = extrema(boundaryaspace)
    )
    plot!(
        vfig,
        boundaryaspace,
        vupper;
        c = beliefscolors[:dark],
        label = L"\mathrm{E}_q[\bar{v}(a,q)]",
    )

    fig = plot(
        wfig,
        vfig;
        layout = (1, 2),
        size = (1100, 420),
        margins = 6Plots.mm,
        link = :y,
    )

    savefig(fig, joinpath(plotpath, "boundary-values.png"))
    fig
end
