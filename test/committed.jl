using Revise
using BenchmarkTools
using FastClosures
using Base.Threads

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false)

includet("../src/utils.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solvers/firm.jl")
includet("../src/solvers/government.jl")

firm = Firm()
government = Government()

# Steady state magnitude check
begin
    n = 301
    A = range(0, 1; length = n)

    fig = plot(A, a -> k(a, firm.δ * a, firm),
        label = L"Steady state investment costs $k(a,, δ a)$",
        xlabel = L"Abatement level $a$",
        ylabel = L"\mathrm{trUSD} / \mathrm{year}",
        color = :darkblue,
    )

    plot!(fig, A, a -> d(a, firm, government) * government.Y,
        label = L"Climate damages $d(a) Y$",
        color = :darkred,
    )
end

# Steady state total costs
begin
    τ₀ = 80 * (3667 / 1_000_000) # ETS price
    
    plot(A, a -> c(a, firm.δ * a, τ₀, firm),
        label = L"Steady state firm's costs $c(a,, δ a, \tau_0)$",
        xlabel = L"Abatement level $a$",
        ylabel = L"\mathrm{trUSD} / \mathrm{year}",
        color = :darkblue,
    )

end

# Firm optimisation
T = 151; θ = 0.01
valuefunction = FirmValue(ones(n, T), ones(n, T) ./ 2);
steadystate!(valuefunction, τ(T, τ₀, θ), A, firm; optimisationstep = 1)

begin
    fig = plot(xlabel = L"Abatemnet level $a_t$", title = L"Tax level $\overline{\tau} = %$(τ₀)$")
	plot!(fig, A, valuefunction.V[:, end]; ylabel = L"Terminal costs $\overline{V} \; [\mathrm{tUSD}]$", yguidefontcolor = :darkblue, c = :darkblue, xlims = (0, 1), linestyle = :dash)
	plot!(twinx(fig), A, valuefunction.Φ[:, end]; ylabel = L"Investment in abatemnet $\overline{\phi}$", yguidefontcolor = :darkred, c = :darkred, ylims = (0, 1), xlims = (0, 1))
end

# Backard induction
backwardinduction!(valuefunction, τ₀, θ, A, firm)
let
	valuefig = contourf(1:T, A, valuefunction.V; xlabel = L"t", ylabel = L"a", title = L"Firm costs $V_t(a)$", camera = (15, 30), opacity = 1, xlims = (1, T), ylims = extrema(A), linewidth = 0.5)

	investmentfig = contourf(1:T, A, valuefunction.Φ; xlabel = L"t", ylabel = L"a", title = L"Investment $\phi_t(a)$", camera = (15, 30), opacity = 1, xlims = (1, T), ylims = extrema(A), linewidth = 0.5)

	plot(valuefig, investmentfig; size = 500 .* (2√2, 1), margins = 10Plots.mm)
end

# Government optimisation
θspace = range(-0.05, 0.05; length = 101)
socialcostcurve = [totalsocialcosts(τ₀, θ, firm, government, valuefunction) for θ in θspace];
plot(θspace, socialcostcurve)
