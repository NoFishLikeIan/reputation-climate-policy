## Modules
using Revise
using UnPack

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using FastClosures
using Optim

using Printf

using Plots, LaTeXStrings
Plots.default(label = false, dpi = 180, size = 350 .* (16/9, 1), margins = 5Plots.mm, linewidth = 2.5)

## Imports
includet("../src/constants.jl")
includet("../src/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/grid.jl")
includet("../src/valuefunction.jl")
includet("../src/boundary.jl")
includet("../src/pfi.jl")

## Setup
firm = Firm()
government = Government()

ns = (51, 31, 31)
signal = Signal(1., 1e-3, gausshermite(ns[3]))

a₀ = 0.
τᶜ = optimize(τᶜ -> w̄(a₀, τᶜ, firm, government, signal), 0., 1., Brent()).minimizer
abatementspace = range(0, 1.25blissabatement(τᶜ, firm, signal), ns[1])
logitspace = gausshermite(ns[2])[1] .* 5
grid = Grid((abatementspace, logitspace))
firmparams = Dict(:maxiter => 100, :valtol => 1e-5, :poltol => 1e-3)
welfareparams = Dict(:maxiter => 100, :valtol => 1e-5, :poltol => 1e-3, :τmax => 2., :τgridpoints => 201)
qspace = pricespace(signal, 0., max(τᶜ, welfareparams[:τmax]), ns[3])

## Value Function
### Initialise firm
firmvalue = FirmValue(grid, qspace)
firmvalue.continuation.V .= 0.
firmvalue.continuation.P .= φ̄(τᶜ, firm, signal)
firmvalue.expost.V .= 0.
firmvalue.expost.P .= φ̄(τᶜ, firm, signal)

### Initialise welfare
welfare = ValueFunction(grid)
for (i, a) in enumerate(abatementspace)
	welfare.V[i, :] .= w̄(a, τᶜ, firm, government, signal)
end
welfare.P .= τ̄(τᶜ, firm, government)


## Check boundaries
if isinteractive()
	wboundaryfig = plot(xlabel = L"Abatement $a$", ylabel = "Trillion of USD", title = "Welfare", ylims = (0, Inf), xlims = extrema(abatementspace))

	plot!(wboundaryfig, abatementspace, a -> w̄(a, τᶜ, firm, government, signal), label = L"\bar{w}", title = "Firm costs", c = :darkgreen)
	plot!(wboundaryfig, abatementspace, a -> w̲(a, firm, government), label = L"\underbar{w}", c = :darkred)

	wboundaryfig
end

if isinteractive()
	vboundaryfig = plot(xlabel = L"Abatement $a$", ylabel = L"Price $q$", ylims = (0, Inf), xlims = extrema(abatementspace), camera = (50, 50))

	surface!(vboundaryfig, abatementspace, qspace, (a, q) -> v̄(a, q, τᶜ, firm, signal))

	vboundaryfig
end

## Iteration
nestedpfi!(firmvalue, welfare, τᶜ, grid, qspace, firm, government, signal; maxiter = 50, valtol = 1e-5, poltol = 1e-5, verbose = 2, firmparams = firmparams, welfareparams = welfareparams)
