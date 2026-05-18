using Revise
using UnPack

using JLD2
using DifferentialEquations, BoundaryValueDiffEq
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

JLD2.@load "data/solutions/continuous-time.jld2" solutions τᶜ signal government firm
solutionsequence = sort(solutions; by = first, rev = true)

function interpolatesolution(φ, solution, τᶜ, government, firm; tol = 1e-6)
    if φ < tol
        return w(0., 0., government, firm), 0.
    elseif φ > 1 - tol
        return w(0., aᶜ(τᶜ, firm), government, firm), 0.
    end

    ℓspace, traj = solution
    u, z = linear_interp(ℓspace, traj, logit(φ))

    return u, z
end

## Plot solutions for continuation convergence of the FS system
solutioncolors = palette(:viridis, length(solutionsequence))

zfig = plot(xlabel = L"Belief $\phi$", xlims = (0.0, 1.0), ylabel = L"z(\phi)")
ufig = plot(xlabel = L"Belief $\phi$", xlims = (0.0, 1.0), ylabel = L"u(\phi)")

for (i, (φstep, ℓspace, trajectory)) in enumerate(solutionsequence)
    label = latexstring("\\varepsilon = ", @sprintf("%.0e", φstep))
    
    φᵢspace = belief.(ℓspace)
    plot!(zfig, φᵢspace, last.(trajectory); color = solutioncolors[i], label)
    plot!(ufig, φᵢspace, first.(trajectory); color = solutioncolors[i], label)
end

solsequencefig = plot(zfig, ufig, size = 400 .* (2√2, 1), margins = 7.5Plots.mm)

savefig(solsequencefig, "figures/faingoldsannikov/solution-sequence.png")

solsequencefig

## Trajectories 
_, solution... = solutionsequence[end]

function beliefevolution(φ, parameters, t)
    solution, τᶜ, signal, government, firm = parameters
    _, z = interpolatesolution(φ, solution, τᶜ, government, firm)

    τ = ηᵉ(φ, z, τᶜ, signal, government, firm) * τᶜ

	ε = signal.ϵ * (τᶜ - τ) / signal.σ
	
    return -φ^2 * (1 - φ) * ε^2
end

φspace = range(0, 1, 1001)
beliefparameters = (solution, τᶜ, signal, government, firm);

φ₀ = 0.5
prob = ODEProblem(beliefevolution, φ₀, (0., 100.), beliefparameters)
sol = solve(prob)

plot(sol)