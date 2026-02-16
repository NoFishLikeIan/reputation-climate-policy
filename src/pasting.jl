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