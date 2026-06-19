using Revise
using UnPack

using JLD2
using DifferentialEquations, BoundaryValueDiffEq, StochasticDiffEq
using FastInterpolations

using LaTeXStrings, Printf
using Colors
using Plots
Plots.default(
    linewidth = 2.5,
    dpi = 180,
    size = (720, 420),
    background_color = :transparent,
)

includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")

includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/solve/equilibrium.jl")
includet("../../src/solve/valuefunction.jl")

includet("colors.jl")

JLD2.@load "data/solutions/static.jld2" solutions τᶜ signal government firm
solutionsequence = sort(solutions; by = first, rev = true)

function interpolatesolution(φ, solution, τᶜ, government, firm; tol = 1e-6)
    if φ < tol
        return w(0., 0., government, firm), 0.
    elseif φ > 1 - tol
        return w(0., aᶜ(τᶜ, firm), government, firm), 0.
    end

    φspace, traj = solution
    u, z = linear_interp(φspace, traj, φ; extrap = ClampExtrap())

    return u, z
end

## Plot solutions for continuation convergence of the FS system
solutioncolors = palette(:viridis, length(solutionsequence))

zfig = plot(xlabel = L"Belief $\phi$", xlims = (0.0, 1.0), ylabel = L"z(\phi)")
ufig = plot(xlabel = L"Belief $\phi$", xlims = (0.0, 1.0), ylabel = L"u(\phi)")

for (i, (φstep, φspace, trajectory)) in enumerate(solutionsequence)
    label = latexstring("\\varepsilon = ", @sprintf("%.0e", φstep))

    plot!(zfig, φspace, last.(trajectory); color = solutioncolors[i], label)
    plot!(ufig, φspace, first.(trajectory); color = solutioncolors[i], label)
end

solsequencefig = plot(zfig, ufig, size = 400 .* (2√2, 1), margins = 7.5Plots.mm)

savefig(solsequencefig, "figures/faingoldsannikov/solution-sequence.png")

solsequencefig

## Trajectories 
_, solution... = solutionsequence[end]

function precision(τ, τᶜ, signal::Signal)
    (τᶜ - τ) * signal.ϵ / signal.σ
end

function beliefevolution(φ, parameters, t)
    solution, τᶜ, signal, government, firm = parameters
    _, z = interpolatesolution(φ, solution, τᶜ, government, firm)

    τ = ηᵉ(φ, z, τᶜ, signal, government, firm) * τᶜ
	
    return -φ^2 * (1 - φ) * precision(τ, τᶜ, signal)^2
end

function beliefvariance(φ, parameters, t)
    solution, τᶜ, signal, government, firm = parameters
    _, z = interpolatesolution(φ, solution, τᶜ, government, firm)

    τ = ηᵉ(φ, z, τᶜ, signal, government, firm) * τᶜ

    return φ * (1 - φ) * precision(τ, τᶜ, signal)
end

monthlytime = 0:(1/12):100
parameters = (solution, τᶜ, signal, government, firm);

prob = SDEProblem(beliefevolution, beliefvariance, 0.99, extrema(monthlytime), parameters; isoutofdomain = (φ, _, _) -> (φ > 1 || φ < 0))
ensembleprob = EnsembleProblem(prob)

φfig = plot(xlabel = "Years", ylabel = L"Belief $\phi_t$", ylims = (0, 1), legendtitle = L"\phi_0")

for φᵢ₀ in 0.1:0.2:0.9
    
    sol = solve(prob, StochasticDiffEq.SRIW2(); u0 = φᵢ₀)
    plot!(φfig, monthlytime, t -> sol(t); label = φᵢ₀)
end

φfig

## Tax path
φ₀ = 0.1
sol = solve(prob; u0 = φ₀)

function taxratio(φ, parameters)
    solution, τᶜ, signal, government, firm = parameters
    _, z = interpolatesolution(φ, solution, τᶜ, government, firm)

    return ηᵉ(φ, z, τᶜ, signal, government, firm)
end

taxpath = map(t -> taxratio(sol(t), parameters), monthlytime)
abatemnetpath = map(t -> begin
    φ = sol(t)
    τ = taxratio(φ, parameters) * τᶜ
    return aᵇ(τ, φ, τᶜ, firm) 
end, monthlytime)

begin
    polfig = plot(monthlytime, taxpath; xlims = extrema(monthlytime), xlabel = "Years", ylabel = L"\tau / \tau^c", c = :black, label = nothing)

    plot!(twinx(), monthlytime, abatemnetpath; ylims = (0, Inf), xlims = extrema(monthlytime), c = :darkgreen, ylabel = L"Abatement $\mathrm{GtCO_2}$", yguidefontcolor = :darkgreen, label = L"a_t", legend = :topright)
end
