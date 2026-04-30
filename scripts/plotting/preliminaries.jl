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

firm = Firm()
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
        mfig;
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
    
    signal_policy_low = 0.6τᶜ
    signal_policy_commit = τᶜ
    signal_qmin = minimum(realisedprice(-4.0, τ, signal) for τ in (signal_policy_low, signal_policy_commit))
    signal_qmax = maximum(realisedprice(4.0, τ, signal) for τ in (signal_policy_low, signal_policy_commit))
    signal_qspace = collect(range(signal_qmin, signal_qmax; length = 500))
    signal_qspace_usd = signal_qspace ./ taxfactor
    signal_density_low = [normalpdf(q, signal.μ * signal_policy_low, signal.σ) * taxfactor for q in signal_qspace]
    signal_density_commit = [normalpdf(q, signal.μ * signal_policy_commit, signal.σ) * taxfactor for q in signal_qspace]
    signal_z_low = [ℓ(q, signal_policy_low, signal_policy_commit, signal) for q in signal_qspace]
    signal_low_mean_usd = signal.μ * signal_policy_low / taxfactor
    signal_commit_mean_usd = signal.μ * signal_policy_commit / taxfactor
    signal_low_mean_z = ℓ(signal.μ * signal_policy_low, signal_policy_low, signal_policy_commit, signal)
    signal_commit_mean_z = ℓ(signal.μ * signal_policy_commit, signal_policy_low, signal_policy_commit, signal)
    signal_z_slope = signal.μ * (signal_policy_commit - signal_policy_low) / signal.σ^2
    signal_zcenter = signal.μ * (signal_policy_low + signal_policy_commit) / 2
    signal_zspace = collect(range(first(signal_z_low), last(signal_z_low); length = length(signal_qspace)))
    signal_qfromz = [signal_zcenter + Δz / signal_z_slope for Δz in signal_zspace]
    signal_z_density_low = [normalpdf(q, signal.μ * signal_policy_low, signal.σ) / abs(signal_z_slope) for q in signal_qfromz]
    signal_z_density_commit = [normalpdf(q, signal.μ * signal_policy_commit, signal.σ) / abs(signal_z_slope) for q in signal_qfromz]

    densityfig = plot(
        signal_qspace_usd,
        signal_density_low;
        c = beliefscolors[:brown],
        xlabel = L"Signal-implied price $q$ (USD/tCO$_2$)",
        legend = :topright,
        label = L"f(q \mid \tau)",
        ylims = (0, maximum(signal_density_low) * 1.05),
        yformatter = _ -> ""
    )
    vline!(densityfig, [signal_low_mean_usd]; c = beliefscolors[:brown], linestyle = :dot, label = L"\mu \tau")

    plot!(
        densityfig,
        signal_qspace_usd,
        signal_density_commit;
        c = beliefscolors[:green],
        label = L"f(q \mid \tau^c)",
    )

    vline!(densityfig, [signal_commit_mean_usd]; c = beliefscolors[:green], linestyle = :dot, label = L"\mu \tau^{c}")

    zfig = plot(
        signal_zspace,
        signal_z_density_low;
        c = beliefscolors[:brown],
        xlabel = L"Reputation change $z^\prime - z$",
        legend = :topright,
        label = L"f(z^\prime-z \mid \tau)",
        ylims = (0, maximum(signal_z_density_low) * 1.05),
        yformatter = _ -> ""
    )
    vline!(zfig, [signal_low_mean_z]; c = beliefscolors[:brown], linestyle = :dot, label = L"\mathbb{E}[z^\prime-z\mid\tau]")

    plot!(
        zfig,
        signal_zspace,
        signal_z_density_commit;
        c = beliefscolors[:green],
        label = L"f(z^\prime-z \mid \tau^c)",
    )

    vline!(zfig, [signal_commit_mean_z]; c = beliefscolors[:green], linestyle = :dot, label = L"\mathbb{E}[z^\prime-z\mid\tau^c]")

    arrowfig = plot(
        xlims = (0, 1),
        ylims = (0, 1),
        framestyle = :none,
        grid = false,
        legend = false,
        ticks = false,
        foreground_color_subplot = :transparent,
        background_color_subplot = :transparent,
    )
    annotate!(arrowfig, 0.5, 0.5, text(L"\Longrightarrow", 34, :black))

    fig = plot(
        densityfig,
        arrowfig,
        zfig;
        layout = @layout([left{0.46w} arrow{0.08w} right{0.46w}]),
        size = (980, 360),
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
    boundary_ylims = (0.0, boundary_ymax)
    valueylabel = "Value [trUSD]"

    wfig = plot(
        boundaryemissionsspace,
        wlower;
        c = beliefscolors[:brown],
        xlabel = emissionslabel,
        ylabel = valueylabel,
        label = L"\underbar{w}(e)",
        legend = :topright,
        ylims = boundary_ylims,
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
        ylims = boundary_ylims
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
        mfig;
        layout = (1, 2),
        size = (1100, 420),
        margins = 6Plots.mm,
        link = :y,
    )

    savefig(fig, joinpath(plotpath, "boundary-values.png"))
    fig
end
