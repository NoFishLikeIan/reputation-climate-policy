using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack
using DifferentialEquations

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/signal.jl")
includet("../src/optimal.jl")

begin
	firm = Firm()
	government = Government()

	τᶜ = committedtax(government, firm)
	
    stackleberg = w(0., 0., government, firm)
	committed = w(τᶜ, aᶜ(τᶜ, firm), government, firm)
end;

begin
	τspace = 0:1:250
	τticks = 0:25:maximum(τspace)
	τticklabels = [L"%$x" for x in 0:25:maximum(τspace)]

	τidxs = sortperm(τticks)

	τticks = τticks[τidxs]
	τticklabels = τticklabels[τidxs]
	τlabel =  L"Carbon tax $\tau \; \textrm{USD} / \textrm{tC}$"
	
	A = 0:0.01:1
	aticks = 0:0.2:1
	aticklabels = [L"%$(floor(Int, 100x)) \%" for x in aticks]
	
	aidxs = sortperm(aticks)
	
	aticks = aticks[aidxs]
	aticklabels = aticklabels[aidxs]
	alabel = L"Abatement rate $a$"
end;

let
	fig = plot(τspace, τ -> wᶜ(τ * taxfactor, government, firm), xlims = extrema(τspace), c = :black, xlabel = τlabel, xticks = (τticks, τticklabels), ylabel = L"Social costs $w^{\mathrm{c}}(\tau) \; \textrm{tUSD} / \textrm{year}$", yguidefontcolor = :black)
	
	vline!(fig, [τᶜ / taxfactor]; c = :black, linestyle = :dashdot, label = L"\tau^{\mathrm{c}}", legend = :topleft)
	
	plot!(twinx(fig), τspace, τ -> aᶜ(τ * taxfactor, firm), xlims = extrema(τspace), ylims = (0, 1.01), c = :darkgreen, ylabel = alabel, yguidefontcolor = :darkgreen, yticks = (aticks, aticklabels))

    savefig(fig, joinpath(plotpath, "committed-carbon-tax.png"))
    fig 
end

signal = Signal()

let a = 1 / 2
	cmin = :black
	cmax = :darkorange

	zs = [0., -0.025, -0.05, -0.1]
	cmap = cgrad([:black, :darkorange])
	
	welfarefig = plot(xlabel = τlabel, xticks = (τticks, τticklabels), legendtitle = L"Reputation $z$", legendtitlefontsize = 9, legendfontsize = 9, ylabel = L"Welfare $\mathcal{L}(\tau; %$(a), z) \textrm{tUSD} / \textrm{year}$", legend = :topleft)
	
	for z in reverse(zs)
		cweight = (z - zs[1]) / (zs[end] - zs[1])
		c = get(cmap, cweight)

		minimizer = bestresponsetax(a, z, signal, government, firm)
		minimum = L(minimizer, a, z, signal, government, firm)
				
		plot!(welfarefig, τspace, τ -> L(τ * taxfactor, a, z, signal, government, firm), label = L"%$z", c = c)
		scatter!(welfarefig, [minimizer / taxfactor], [minimum], c = :black)
		annotate!(welfarefig, minimizer / taxfactor, minimum, text(L"\tau = %$(round(minimizer / taxfactor, digits = 2))", 10, :bottom))
	end


	bestresponsefig = plot(ylabel = τlabel, yticks = (τticks, τticklabels), xlabel = L"Reputational weight $z$")

	τ̄ = bestresponsetax(a, Inf, signal, government, firm) / taxfactor

	zspace = range(-0.02, 0., 101)

	plot!(bestresponsefig, zspace, z -> bestresponsetax(a, z, signal, government, firm) / taxfactor; c = :black, ylims = (0, τᶜ  / taxfactor), xlims = extrema(zspace), xflip = true)

	hline!(bestresponsefig, [τ̄ ]; c = :black, linestyle = :dash, label = L"\lim_{z \to \infty} \tau")
	hline!(bestresponsefig, [τᶜ  / taxfactor]; c = :black, linestyle = :dot, label = L"\tau^{\mathrm{c}}")

	plot(welfarefig, bestresponsefig; size = (900, 400), margins = 5Plots.mm)
end