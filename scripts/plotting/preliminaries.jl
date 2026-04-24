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

signal_price = qspace ./ taxfactor
signal_shift_nc = [ℓ(q, 0.5τᶜ, τᶜ, signal) for q in qspace]
signal_shift_commit = [ℓ(q, τᶜ, τᶜ, signal) for q in qspace]

## Plots
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
    boundary_aspace = collect(range(0.0, aspace[end]; length = 250))
    wlower = [w̲(a, firm, government) for a in boundary_aspace]
    wupper = [w̄(a, τᶜ, firm, government, signal) for a in boundary_aspace]
    mlower = zeros(length(boundary_aspace))
    mupper = [ψ̄(a, τᶜ, firm, signal) for a in boundary_aspace]
    boundary_ymax = maximum((maximum(wlower), maximum(wupper), maximum(mlower), maximum(mupper)))
    boundary_ylims = (0.0, boundary_ymax)
    valueylabel = "Value [trUSD]"

    wfig = plot(
        boundary_aspace,
        wlower;
        c = beliefscolors[:brown],
        xlabel = alabel,
        ylabel = valueylabel,
        label = L"\underbar{w}(a)",
        legend = :topright,
        ylims = boundary_ylims,
    )
    plot!(
        wfig,
        boundary_aspace,
        wupper;
        c = beliefscolors[:dark],
        label = L"\bar{w}(a)",
    )

    mfig = plot(
        boundary_aspace,
        mlower;
        c = beliefscolors[:brown],
        xlabel = alabel,
        ylabel = valueylabel,
        label = L"\underbar{m}(a)",
        legend = :topright,
        ylims = boundary_ylims
    )
    plot!(
        mfig,
        boundary_aspace,
        mupper;
        c = beliefscolors[:dark],
        label = L"\bar{m}(a)",
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
