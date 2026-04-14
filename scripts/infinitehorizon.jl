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
ns = (20, 15, 15)
abatementspace = range(0, 10, ns[1])
logitspace, _ = gausshermite(ns[2])
stategrid = Grid((abatementspace, logitspace))

firm = Firm()
government = Government()
signal = Signal(1., 1., gausshermite(ns[3]))

## Value Function
welfare = ValueFunction(stategrid); welfare.V .= 1.; welfare.P .= 0.
firmvalue = ValueFunction(stategrid, signal); firmvalue.V .= 1.; firmvalue.P .= 0.

## Iteration
τᶜ = 0.25
τmax = 1.5

nestedpfi!(
	firmvalue,
	welfare,
	τᶜ,
	τmax,
	signal,
	stategrid,
	firm,
	government;
	maxiter = 50,
	valtol = 1e-5,
	poltol = 1e-5,
	verbose = 1,
	firmkwargs = (; maxiter = 100, valtol = 1e-5, poltol = 1e-5),
	govkwargs = (; maxiter = 100, valtol = 1e-5, poltol = 1e-5)
)
