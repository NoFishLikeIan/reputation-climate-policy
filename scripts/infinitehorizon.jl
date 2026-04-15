## Modules
using Revise

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using FastClosures
using Optim

using Printf

## Imports
includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/signal.jl")

includet("../src/grid.jl")
includet("../src/valuefunction.jl")
includet("../src/pfi.jl")

## Setup
ns = (20, 31, 15)
zmax = 10.0
abatementspace = range(0, 10, ns[1])
logitspace = collect(range(-zmax, zmax, length = ns[2]))
stategrid = Grid((abatementspace, logitspace))

firm = Firm()
government = Government()
signal = Signal(1., 1., gausshermite(ns[3]))

τᶜ = 0.25

## Value Function
welfare = ValueFunction(stategrid)
firmvalue = ValueFunction(stategrid, signal)

welfare.V .= 1.
for (j, z) in enumerate(logitspace)
	welfare.V[:, j] .+= 1e-3 * z
	welfare.P[:, j] .= max.(zero(eltype(welfare.P)), τᶜ .+ 1e-3 * z)
end

signalspace, _ = signal.space
for (j, z) in enumerate(logitspace)
	for (k, ξ) in enumerate(signalspace)
		s = signal.μ * τᶜ + sqrt2 * signal.σ * ξ
		firmvalue.V[:, j, k] .= 1 .- 1e-2 .* abatementspace .* s .+ 1e-3 * z
		firmvalue.P[:, j, k] .= max.(zero(eltype(firmvalue.P)), 1e-2 .+ 5e-4 * z .+ 5e-4 * ξ)
	end
end

## Iteration
innerparams = Dict(:maxiter => 100, :valtol => 1e-5, :poltol => 1e-3)

nestedpfi!(firmvalue, welfare, τᶜ, signal, stategrid, firm, government; maxiter = 50, valtol = 1e-5, poltol = 1e-5, verbose = 2, firmparams = innerparams, welfareparams = innerparams)
