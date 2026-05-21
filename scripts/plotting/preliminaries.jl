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
includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/primitives/grid.jl")
includet("../../src/solve/valuefunction.jl")
includet("../../src/solve/boundary.jl")

const taxfactor = scctotax

firm = StaticFirm()
government = Government()

σestimated = 1.1;
signal = Signal(1., σestimated * scctotax, 101)

## Compute figure values
τᶜ = optimize(τ -> w̄(e₀, τ, firm, government, signal), 0.0, 1.0, Brent()).minimizer
φstar = φ̄(τᶜ, firm, signal)

τspace_usd = collect(range(0.0, 250.0; length = 251))
τspace = τspace_usd .* taxfactor
emissionsspace = collect(range(0.0, e₀; length = 250))
φspace = collect(range(0.0, 1.5 * max(φstar, 1e-6); length = 250))
ξspace = collect(range(-3.0, 3.0; length = 250))
qspace = realisedprice.(ξspace, τᶜ, Ref(signal))
zspace = collect(range(-0.25, 0.05; length = 250))
zs = [0.0, -0.025, -0.05, -0.10]

τlabel = L"Carbon tax $\tau$ (USD/tCO$_2$)"
emissionslabel = L"Emissions $e$ [GtCO$_2$ / year]"
φlabel = L"Abatement investment $\phi$ [GtCO$_2$ / year]"
zlabel = L"Reputation $z$"

normalpdf(x, μ, σ) = @inline exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

signal_price = qspace ./ taxfactor
signal_shift_nc = [ℓ(q, 0.5τᶜ, τᶜ, signal) for q in qspace]
signal_shift_commit = [ℓ(q, τᶜ, τᶜ, signal) for q in qspace]

# Plots
## Compare boundaries by δ
let
    deltavalues = [0.0, 1e-2]
    deltafirms = [Firm(δ = δ) for δ in deltavalues]
    deltalabels = ["0", "0.025"]
    deltalinestyles = [:solid, :dot]
    lowercolor = beliefscolors[:brown]
    uppercolor = beliefscolors[:dark]
    boundaryemissionsspace = emissionsspace
    committedtaxes = [
        optimize(τ -> w̄(e₀, τ, deltafirm, government, signal), 0.0, 1.0, Brent()).minimizer
        for deltafirm in deltafirms
    ]

    wlowers = [
        [w̲(emissions, deltafirm, government) for emissions in boundaryemissionsspace]
        for deltafirm in deltafirms
    ]
    wuppers = [
        [w̄(emissions, committedtax, deltafirm, government, signal) for emissions in boundaryemissionsspace]
        for (deltafirm, committedtax) in zip(deltafirms, committedtaxes)
    ]
    muppers = [
        [m̄(emissions, committedtax, deltafirm, signal) for emissions in boundaryemissionsspace]
        for (deltafirm, committedtax) in zip(deltafirms, committedtaxes)
    ]

    valuemax = maximum((
        maximum(maximum, wlowers),
        maximum(maximum, wuppers),
        maximum(maximum, muppers),
    ))
    valueylims = (0.0, valuemax)
    valueylabel = "Value [trUSD]"

    wfig = plot(
        xlabel = emissionslabel,
        ylabel = valueylabel,
        legend = :topright,
        ylims = valueylims,
    )
    mfig = plot(
        xlabel = emissionslabel,
        ylabel = valueylabel,
        legend = :topright,
        ylims = valueylims,
    )

    for (i, deltalabel) in enumerate(deltalabels)
        plot!(
            wfig,
            boundaryemissionsspace,
            wlowers[i];
            c = lowercolor,
            linestyle = deltalinestyles[i],
            label = latexstring("\\underbar{w}(e), \\delta = ", deltalabel),
        )
        plot!(
            wfig,
            boundaryemissionsspace,
            wuppers[i];
            c = uppercolor,
            linestyle = deltalinestyles[i],
            label = latexstring("\\bar{w}(e), \\delta = ", deltalabel),
        )

        plot!(
            mfig,
            boundaryemissionsspace,
            zeros(length(boundaryemissionsspace));
            c = lowercolor,
            linestyle = deltalinestyles[i],
            label = latexstring("\\underbar{m}(e), \\delta = ", deltalabel),
        )
        plot!(
            mfig,
            boundaryemissionsspace,
            muppers[i];
            c = uppercolor,
            linestyle = deltalinestyles[i],
            label = latexstring("\\bar{m}(e), \\delta = ", deltalabel),
        )
    end

    fig = plot(
        wfig,
        vfig;
        layout = (1, 2),
        size = (1100, 420),
        margins = 6Plots.mm,
        link = :y,
    )

    savefig(fig, joinpath(plotpath, "boundary-values-delta.png"))
    fig
end


## Boundary Values
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
    boundaryemissionsspace = emissionsspace
    wlower = [w̲(emissions, firm, government) for emissions in boundaryemissionsspace]
    wupper = [w̄(emissions, τᶜ, firm, government, signal) for emissions in boundaryemissionsspace]
    mlower = zeros(length(boundaryemissionsspace))
    mupper = [m̄(emissions, τᶜ, firm, signal) for emissions in boundaryemissionsspace]
    boundary_ymax = maximum((maximum(wlower), maximum(wupper), maximum(mlower), maximum(mupper)))
    boundary_ylims = (-0.05, boundary_ymax)
    valueylabel = "Value [trUSD]"

    wfig = plot(
        boundaryemissionsspace,
        wlower;
        c = beliefscolors[:brown],
        xlabel = emissionslabel,
        ylabel = valueylabel,
        label = L"\underbar{w}(e)",
        legend = :topright,
        ylims = boundaryylims,
        xlims = extrema(boundaryaspace)
    )
    plot!(
        wfig,
        boundaryemissionsspace,
        wupper;
        c = beliefscolors[:dark],
        label = L"\bar{w}(e)",
    )

    mfig = plot(
        boundaryemissionsspace,
        mlower;
        c = beliefscolors[:brown],
        xlabel = emissionslabel,
        ylabel = valueylabel,
        label = L"\underbar{m}(e)",
        legend = :topright,
        ylims = boundaryylims,
        xlims = extrema(boundaryaspace)
    )
    plot!(
        mfig,
        boundaryemissionsspace,
        mupper;
        c = beliefscolors[:dark],
        label = L"\bar{m}(e)",
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
