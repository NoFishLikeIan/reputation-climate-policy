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
    n = 501
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
    τ₀ = (83 * 3667 / 1_000_000) # ETS price
    
    plot(A, a -> c(a, firm.δ * a, τ₀, firm),
        label = L"Steady state firm's costs $c(a,, δ a, \tau_0)$",
        xlabel = L"Abatement level $a$",
        ylabel = L"\mathrm{trUSD} / \mathrm{year}",
        color = :darkblue,
    )

end


T = 501
valuefunction = FirmValue(ones(n, T), ones(n, T) ./ 2);
tolerance = Error(1e-10, 1e-10)

howard!(valuefunction, τ₀, A, firm, tolerance, tolerance; verbose = true)

begin
    fig = plot(xlabel = L"Abatemnet level $a_t$", title = L"Tax level $\overline{\tau} = %$(τ₀)$")
	plot!(fig, A, valuefunction.V[:, T]; ylabel = L"Terminal costs $\overline{V} \; [\mathrm{tUSD}]$", yguidefontcolor = :darkblue, c = :darkblue, xlims = (0, 1), linestyle = :dash)
	plot!(twinx(fig), A, valuefunction.Φ[:, T]; ylabel = L"Investment in abatemnet $\overline{\phi}$", yguidefontcolor = :darkred, c = :darkred, ylims = (0, 1), xlims = (0, 1))
end