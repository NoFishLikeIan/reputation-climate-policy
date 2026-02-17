using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack

using Roots
using Optim
using ForwardDiff, DifferentiationInterface, DifferentialEquations

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/valuefunction"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/primitives/constants.jl")
includet("../src/primitives/firm.jl")
includet("../src/primitives/government.jl")
includet("../src/primitives/signal.jl")
includet("../src/primitives/optimal.jl")
includet("../src/hjb.jl")

includet("../src/pasting.jl")

# Example call with original values
begin
	firm = Firm()
    signal = Signal()
	government = Government()
	ε = 1e-3
end;


cs = Float64[]
for (i, δ) in enumerate(δs)
	@printf "Solving for δ = %.2f\n" δ
	
	government = Government(δ = δ)
	model = (signal, government, firm)
	optparameters = (model, ε)

	shootingsol = optimize(cₗ -> shootingerror(cₗ, optparameters), -100_000., 0.)

	if shootingsol.stopped_by[:converged]
		@printf "Optimization converged with minimum %.2f\n" shootingsol.minimum
	else
		@warn "Optimization failed"
	end

	push!(cs, shootingsol.minimizer)
end

let
	ufig = plot( xlims = (0, 1), ylabel =  L"u(\phi)")
	zfig = plot( xlims = (0, 1), ylabel =  L"z(\phi)", xlabel =  L"\phi")
	
	for (i, δ) in enumerate(δs)
		model = (signal, Government(δ = δ), firm)
		x = leftinit(cs[i], ε, model)

		prob = ODEProblem{false}(F, x, (ε, 1 - ε), model)
		sol = solve(prob, Rodas4P()) 

		unit = range(ε, 1 - ε, 1001)
		ufn = φ -> sol(φ)[1]
		zfn = φ -> sol(φ)[2]

		plot!(ufig, unit, ufn; label = L"\delta = %$δ", linewidth = 2)
		plot!(zfig, unit, zfn; label = L"\delta = %$δ", linewidth = 2)
	end
	
	plot(ufig, zfig; layout = (2, 1), link = :x, size = 450 .* (√2, 1.))
end