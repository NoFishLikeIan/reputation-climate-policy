using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack

using StaticArrays, DifferentialEquations
using Optimization, OptimizationOptimJL
using ForwardDiff, DifferentiationInterface

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/signal.jl")
includet("../src/optimal.jl")
includet("../src/hjb.jl")

function leftinit(cₗ, ε, model)
	θ = zero(ε)
	government = model[2]
	r = government.r
	
	κfn = Base.Fix{3}(κ², model)
	wfn = Base.Fix{3}(wᵒ, model)

	κ₀ = κfn(θ, θ)
	κ′₀ = ForwardDiff.derivative(Base.Fix{2}(κfn, θ), θ)
	
	w₀ = wfn(θ, θ)
	w′₀ = ForwardDiff.derivative(Base.Fix{2}(wfn, θ), θ)
	
	m = (1 + √(1 + 8r * κ₀)) / 2
	uₘ = cₗ
	uₘ₊₁ = (m + r * κ′₀ / m) * uₘ + (r / m) * (κ₀ * w′₀ + w₀ * κ′₀)
	
	uₗ = w₀ + uₘ * ε^m + uₘ₊₁ * ε^(m + 1)
	zₗ = (m * uₘ * ε^m) / r + ((m + 1) * uₘ₊₁ - m * uₘ) * ε^(m + 1) / r
	
	return SVector(uₗ, zₗ)
end
function rightinit(cᵣ, ε, model)
	ι = one(ε)
	government = model[2]
	r = government.r
	
	κfn = Base.Fix{3}(κ², model)
	wfn = Base.Fix{3}(wᵒ, model)

	κ₁ = κfn(ι, ι)
	κ′₁ = ForwardDiff.derivative(Base.Fix{2}(κfn, ι), ι)
	
	w₁ = wfn(ι, ι)
	w′₁ = ForwardDiff.derivative(Base.Fix{2}(wfn, ι), ι)
	
	n = (1 + √(1 + 8r * κ₁)) / 2
	uₙ = cᵣ
	uₙ₊₁ = (n + r * κ′₁ / n) * uₙ + (r / n) * (κ₁ * w′₁ + w₁ * κ′₁)
	
	uᵣ = w₁ + uₙ * ε^n + uₙ₊₁ * ε^(n + 1)
	zᵣ = - ((n * uₙ * ε^n) / r + ((n + 1) * uₙ₊₁ - n * uₙ) * ε^(n + 1) / r)
	
	return SVector(uᵣ, zᵣ)
end

function pastingerror(c, parameters)
	model, ε, φₘ = parameters
	cₗ, cᵣ = c	
	xₗ = leftinit(cₗ, ε, model)
	xᵣ = rightinit(cᵣ, ε, model)

	leftprob = ODEProblem{false}(F, xₗ, (ε, φₘ), model)
	leftsol = solve(leftprob, Rodas4P(); save_everystep = false, save_end = true, save_start = false) 
	
	rightprob = ODEProblem{false}(F, xᵣ, (1 - ε, φₘ), model)
	rightsol = solve(rightprob, Rodas4P(); save_everystep = false, save_end = true, save_start = false)
	
	return sum(abs2, leftsol.u[1] - rightsol.u[1])
end

# Example call with original values
begin
	firm = Firm()
    signal = Signal()
	ε = 1e-4
	φₘ = 0.5
end;

δs = [0., 0.01, 0.05]

cs = MVector{2, Float64}[]
for (i, δ) in enumerate(δs)
	@printf "Solving for %.2f" δ
	c₀ = i > 1 ? cs[i - 1] : MVector(-500., 6000.)
	
	objfn = SciMLBase.OptimizationFunction(pastingerror,AutoForwardDiff())

	government = Government(δ = δ)
	model = (signal, government, firm)

	pastingproblem = OptimizationProblem(objfn, x₀, (model, ε, φₘ))
	pastingsol = solve(pastingproblem, BFGS(); iterations = 2_000)

	if !SciMLBase.successful_retcode(pastingsol.retcode)
		@warn "Optimization failed for δ = $δ with retcode $(pastingsol.retcode)"
	end

	push!(cs, pastingsol.u)
end

let
	ufig = plot()
	
	for (i, δ) in enumerate(δs)
		model_δ = (signal, Government(δ = δ), firm)
		cₗ, cᵣ = cs[i]
		xₗ = leftinit(cₗ, ε, model_δ)
		xᵣ = rightinit(cᵣ, ε, model_δ)

		leftprob = ODEProblem{false}(F, xₗ, (ε, φₘ), model_δ)
		leftsol = solve(leftprob, Rodas4P()) 

		rightprob = ODEProblem{false}(F, xᵣ, (1 - ε, φₘ), model_δ)
		rightsol = solve(rightprob, Rodas4P())

		unit = range(ε, 1 - ε, 101)
		ufn = φ -> φ < φₘ ? leftsol(φ)[1] : rightsol(φ)[1]
		zfn = φ -> φ < φₘ ? leftsol(φ)[2] : rightsol(φ)[2]

		plot!(ufig, unit, ufn; xlims = (0, 1), label = L"\delta = "*string(δ))
	end
	
	ylabel!(ufig, L"u(\phi)")
	xlabel!(ufig, L"\phi")
end