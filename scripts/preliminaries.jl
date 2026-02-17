using Revise
using BenchmarkTools

using UnPack
using Roots
using Optim

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/primitives/constants.jl")
includet("../src/primitives/firm.jl")
includet("../src/primitives/government.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/optimal.jl")
includet("../src/hjb.jl")

includet("../src/pasting.jl")

# Example call with original values
begin
	firm = Firm(δ = Inf)
    signal = Signal()
	government = Government()
end;

solcommitted = Optim.optimize(τ -> wᶜ(τ, government, firm), 0., 300 * taxfactor)
τᶜ = solcommitted.minimizer
committedwelfare = solcommitted.minimum

let
	zspace = range(0, 150, 1001)

	equilibriumtaxfig = plot(xlabel = L"Reputation weight $z$", ylabel = L"Equilibrium tax $\tau^*$", c = :black, linewidth = 2.5, xlims = (0., Inf), ylims = (0., Inf), legendtitle = L"\varphi", legend = :topright)
	
	φspace = 0:0.2:1.
	cmap = palette([:darkorange, :darkblue], length(φspace))
	
	for (i, φ) ∈ enumerate(φspace)
		plot!(equilibriumtaxfig, zspace, z -> equilibriumtax(z, φ, signal, firm) / taxfactor; label = φ, c = cmap[i])
	end

	hline!(equilibriumtaxfig, [equilibriumtax(Inf, 0.5, signal, firm) / taxfactor]; c = :black)
	hline!(equilibriumtaxfig, [τᶜ / taxfactor]; c = :black, linestyle = :dash)

	equilibriumtaxfig
end

