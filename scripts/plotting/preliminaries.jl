using Revise
using UnPack

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions

using Optim
using LaTeXStrings, Printf

using Plots
default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)

plotpath = "paper/figures/preliminaries"
if !ispath(plotpath)
    mkpath(plotpath)
end

includet("../../src/constants.jl")
includet("../../src/signal.jl")
includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/grid.jl")
includet("../../src/valuefunction.jl")
includet("../../src/boundary.jl")
includet("../../src/boundary.jl")

const taxfactor = scctotax

firm = Firm()
government = Government()
signal = Signal(1.0, 0.05, 41)

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

committed_welfare = [w̄(a0, τ, firm, government, signal) for τ in τspace]
committed_investment = [φ̄(τ, firm, signal) for τ in τspace]
committed_bliss = [blissabatement(τ, firm, signal) for τ in τspace]

firm_cost = [c(φ, firm) for φ in φspace]
firm_marginal_cost = [cᵩ(φ, firm) for φ in φspace]
emissions = [e(a, firm) for a in aspace]
damages = [d(e(a, firm), government) for a in aspace]

signal_price = qspace ./ taxfactor
signal_shift_nc = [ℓ(q, 0.5τᶜ, τᶜ, signal) for q in qspace]
signal_shift_commit = [ℓ(q, τᶜ, τᶜ, signal) for q in qspace]

function L(τ, a, z, signal::Signal, government::Government, firm::Firm)
    μgap = signal.μ * (τᶜ - τ)
    return c(φ̄(τᶜ, firm, signal), firm) + d(e(a, firm), government) - z * (signal.μ * τ) * μgap / signal.σ^2
end

bestresponsetax(a, z, signal::Signal, government::Government, firm::Firm) = optimize(
    τ -> L(τ, a, z, signal, government, firm), 0.0, 2τᶜ, Brent()
).minimizer

welfare_curves = Dict(z => [L(τ, 0, z, signal, government, firm) for τ in τspace] for z in zs)
best_response = [bestresponsetax(0, z, signal, government, firm) / taxfactor for z in zspace]
τ∞ = bestresponsetax(0, -1e6, signal, government, firm) / taxfactor

let
    fig = plot(
        τspace_usd,
        committed_welfare;
        c = :black,
        xlabel = τlabel,
        ylabel = L"Value / policy",
        legend = :topleft,
        label = L"\bar w(a_0,\tau)",
    )

    plot!(fig, τspace_usd, committed_investment; c = :darkgreen, label = L"\bar \phi(\tau)")
    plot!(fig, τspace_usd, committed_bliss; c = :darkorange, label = L"a^{\mathrm{bliss}}(\tau)")
    vline!(fig, [τᶜ / taxfactor]; c = :gray, linestyle = :dashdot, label = L"\tau^{\mathrm c}")
	savefig(fig, joinpath(plotpath, "committed-abatement.svg"))
    	fig
end

let
    fig = plot(
        φspace,
        firm_cost;
        c = :darkred,
        xlabel = φlabel,
        ylabel = "Firm adjustment cost [trUSD per year]",
        legend = :topleft,
        label = L"c(\phi) = \kappa \phi + \frac{\nu}{2} \phi^2",
		xlims = (0, Inf), ylims = (0, Inf), linewidth = 3
    )
	savefig(fig, joinpath(plotpath, "firm-costs.svg"))
    fig
end

let
    fig = plot(
        aspace,
        damages;
        c = :black,
        xlabel = alabel,
        ylabel = "Climate damages [trUSD]",
        legend = :topright,
        label = L"d(e_0 - a)",
		xlims = (0, Inf), ylims = (0, Inf)
    )
	savefig(fig, joinpath(plotpath, "emissions-damages.svg"))
    fig
end

let
    fig = plot(
        ξspace,
        signal_price;
        c = :black,
        xlabel = L"Innovation node $\xi$",
        ylabel = "Signal-implied object",
        legend = :topleft,
        label = L"q(\xi,\tau^{\mathrm c}) \textrm{ in USD/tCO}_2",
    )
    plot!(fig, ξspace, signal_shift_nc; c = :darkorange, label = L"\ell(q,0.5\tau^{\mathrm c},\tau^{\mathrm c})")
    plot!(fig, ξspace, signal_shift_commit; c = :gray, label = L"\ell(q,\tau^{\mathrm c},\tau^{\mathrm c})")
	savefig(fig, joinpath(plotpath, "signal-mapping.svg"))
    fig
end

let
    fig = plot(
        τspace_usd,
        welfare_curves[0.0];
        c = :black,
        xlabel = τlabel,
        ylabel = "Government objective",
        legend = :bottomright,
        label = L"z=0",
    )
    plot!(fig, τspace_usd, welfare_curves[-0.025]; c = :steelblue, label = L"z=-0.025")
    plot!(fig, τspace_usd, welfare_curves[-0.05]; c = :darkgreen, label = L"z=-0.05")
    plot!(fig, τspace_usd, welfare_curves[-0.10]; c = :darkorange, label = L"z=-0.10")
	savefig(fig, joinpath(plotpath, "reputation-objective.svg"))
    fig
end

let
    fig = plot(
        zspace,
        best_response;
        c = :black,
        xlabel = zlabel,
        ylabel = L"Best-response tax (USD/tCO$_2$)",
        legend = :topleft,
        xflip = true,
        label = L"\tau^*(a,z)",
    )
    hline!(fig, [τ∞]; c = :darkorange, linestyle = :dash, label = L"\lim_{z\to -\infty}\tau^*(a,z)")
    hline!(fig, [τᶜ / taxfactor]; c = :gray, linestyle = :dot, label = L"\tau^{c}")
	savefig(fig, joinpath(plotpath, "best-response-tax.svg"))
    fig
end
