## Optimal tax
committedtaxresidual(τ, (x, model)) = committedtaxresidual(τ, x, model)
function committedtaxresidual(τ, x, model::AbstractModelParameters)
    @unpack household, firm, climate = model
    m, _, _, λₘ, _, λᵨ = x # State, co-state, cumulative costs (z, λ, u)

    wage = ω(τ, firm)

    return firm.A * n′(wage, household) * ω′(τ, firm) * (d(m, climate) + firm.η * ( λₘ - τ )) - household.r * λᵨ
end
function committedtax(x::TX, model::AbstractModelParameters) where {T, TX <: AbstractVector{T}}
    𝟎 = zero(T)

    taxupperbound = (1 - √eps(T)) / model.firm.η
    foc = Base.Fix2(committedtaxresidual, (x, model))

    fl = foc(𝟎)
    fu = foc(taxupperbound)

    if fl ≥ 0
        return 𝟎
    elseif fu ≤ 0
        taxupperbound
    else
        return Roots.find_zero(foc, (𝟎, taxupperbound), Roots.Brent())
    end
end
function committedtax(x::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    @unpack household, firm, climate = model
    m, _, _, λₘ, _, λᵨ = x # State, co-state, cumulative costs (z, λ, u)

    interiortax = (d(m, climate) / firm.η) + λₘ + household.r * household.ν * λᵨ / (firm.A * firm.η)^2

    return clamp(interiortax, 0, inv(firm.η))
end

## State-Costate system
initialguess(model) = initialguess(SA.SVector, model)
function initialguess(TV, model::AbstractModelParameters)
    @unpack household, firm, climate = model

    q₀ = τ₀
    λₘ₀ = τ₀ - d(climate.m₀, climate) / firm.η
    λₐ₀ = -q₀ / household.r
    λᵨ₀ = zero(λₐ₀)

    return TV(climate.m₀, firm.a₀, q₀, λₘ₀, λₐ₀, λᵨ₀)
end

function driftcommitted!(dx, x, model::AbstractModelParameters, t)
    @unpack household, firm, climate = model 
    m, a, q, λₘ, λₐ, λᵨ = x # State, co-state, cumulative costs (z, λ, u)
    
    τᶜ = committedtax(x, model)
    wage = ω(τᶜ, firm)
    labour = n(wage, household)

    dm = firm.η * firm.A * labour - a
    da = e(labour, a, firm) > 0 ? α(q, a, firm) : zero(a)
    dq = household.r * (q - τᶜ + firm.κ * da)

    dλₘ = household.r * λₘ - firm.A * labour * d′(m, climate)
    dλₐ = (household.r + firm.κ / firm.ξ) * λₐ + λₘ + (household.r*firm.κ^2 / firm.ξ) * λᵨ + (a * firm.κ^2 / firm.ξ)
    dλᵨ = -firm.κ / firm.ξ * λᵨ - λₐ / (household.r * firm.ξ) - q / (household.r^2 * firm.ξ)

    dx[1] = dm
    dx[2] = da
    dx[3] = dq
    dx[4] = dλₘ
    dx[5] = dλₐ
    dx[6] = dλᵨ
    
    return dx
end

## Truncation
function ρd̄′(t, (τᶜ, x, model))
    m, a = @view x[1:2]
    wage = ω(τᶜ, model.firm)
    labour = n(wage, model.household)
    mₜ = m + e(labour, a, model.firm) * t

    return d′(mₜ, model.climate) * exp(-model.household.r * t)
end
function tρd̄′(t, (τᶜ, x, model))
    ρd̄′(t, (τᶜ, x, model)) * t
end

function ρd̄(t, (τᶜ, x, model))
    m, a = @view x[1:2]
    wage = ω(τᶜ, model.firm)
    labour = n(wage, model.household)
    mₜ = m + e(labour, a, model.firm) * t

    return d(mₜ, model.climate) * exp(-model.household.r * t)
end

"Terminal gradient of the value function"
function ∇v̄(x, model::AbstractModelParameters)
    @unpack household, firm, climate = model
    m, a, q = @view x[1:3]

    wage = ω(q, firm)

    output = firm.A * n(wage, household)
    output′ = firm.A * n′(wage, household) * ω′(q, firm)
    emissions = firm.η * output - a

    ecds = climate.γ * climate.ζ^2

    aₘ = ecds * m^2 / 2
    bₘ = household.r + m * ecds * emissions
    cₘ = ecds * emissions^2 / 2

    Jₘ = J(aₘ, bₘ, cₘ)
    Gₘ = G(aₘ, bₘ, cₘ)

    ∂ₘv = output * ecds * (m * Jₘ + emissions * Gₘ)
    ∂ₐv = -output * ecds * (m * Gₘ + emissions * L(aₘ, bₘ, cₘ))
    ∂ᵨv = output′ * ((1 / household.r) - Jₘ - firm.η * ∂ₐv - firm.η * q / household.r)

    return (∂ₘv, ∂ₐv, ∂ᵨv)
end


## Outer Optimization
optimisationbounds(model) = optimisationbounds(SA.SVector, model)
function optimisationbounds(TV, model::AbstractModelParameters; λmax = Inf)
    @unpack household, firm, climate = model
  
    q₀ = household.r * c(firm.a₀, firm)

    lb = TV(climate.m₀, firm.a₀, q₀, -λmax, -λmax, -λmax)
    ub = TV(climate.m₀, firm.a₀, λmax, λmax, λmax, λmax)

    return lb, ub
end

function initialcondition(x₀::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    initialcondition!(Vector{Float64}(undef, 3), x₀, model)
end
function initialcondition!(res, x₀, model::LinearLabourModelParameters)
    res[1] = x₀[1] - model.climate.m₀
    res[2] = x₀[2] - model.firm.a₀
    res[3] = τ₀ - committedtax(x₀, model)

    return res
end

function terminalcondition(x̄::TX, model::AbstractModelParameters) where {T, TX <: AbstractVector{T}}
    terminalcondition!(Vector{Float64}(undef, 3), x̄, model)
end
function terminalcondition!(res, x̄, model::AbstractModelParameters)
    @unpack firm, household = model
    ∂ₘv, ∂ₐv, ∂ᵨv = ∇v̄(x̄, model)
    _, a, q, λₘ, λₐ, λᵨ = x̄ # State, co-state, cumulative costs (z, λ, u)̄

    wage = ω(q, firm)
    labour = n(wage, household)
    output′ = firm.η * firm.A * n′(wage, household) * ω′(q, firm)
    
    res[1] = e(labour, a, firm) # No emissions
    res[2] = ∂ₘv - λₘ
    res[3] = (λᵨ - ∂ᵨv) + output′ * (λₐ - ∂ₐv)

    return res
end

"Drift of state co-state system with welfare as last variable"
function driftwelfarecommitted!(dz, z, model, t)
    # State co-state drift
    dx = @view dz[1:6]
    x = @view z[1:6]

    driftcommitted!(dx, x, model, t)

    # Welfare costs drift
    @unpack household, firm, climate = model
    da, dq = @view dx[2:3]
    m, a, q = @view x[1:3]
    τᶜ = q - dq / household.r + firm.κ * da

    dw = exp(-household.r * t) * w(τᶜ, m, a, da, household, firm, climate)

    dz[7] = dw

    return dz

end

const defbvpalg = BVP.MIRK4()
const defodealg = ODE.AutoTsit5(ODE.Rosenbrock23())

function objectivewelfare(T, parameters)
    bvproblem, welfareproblem, odeproblem, model, normalisedstep, bvpalg, odealg = parameters
    return objectivewelfare(T, bvproblem, welfareproblem, odeproblem, model; normalisedstep, bvpalg, odealg)
end
function objectivewelfare(T::TX, bvproblem::BVP.BVProblem, welfareproblem::ODE.ODEProblem, odeproblem::ODE.ODEProblem, model::AbstractModelParameters; normalisedstep = 0.1, bvpalg = defbvpalg, odealg = defodealg) where TX
    odesolguess = ODE.solve(odeproblem, odealg; tspan = T)
    bvpsolution = BVP.solve(bvproblem, bvpalg; u0 = odesolguess, dt = normalisedstep * T, tspan = (0., T))

    if !SciMLBase.successful_retcode(bvpsolution)
        @warn "Unsuccessful BV solution with T = $T. Retcode = $(bvpsolution.retcode)"
    end

    x₀ = bvpsolution.u[1]
    z₀ = SA.MVector{7, TX}(x₀..., 0.)
    welfaresolution = ODE.solve(welfareproblem, odealg; u0 = z₀, save_end = true, save_everystep = false, save_start = false, tspan = T)

    z̄ = only(welfaresolution.u)
    m̄, ā, q̄ = @view z̄[1:3]

    ū = w(q̄, m̄, ā, 0, model.household, model.firm, model.climate) / model.household.r
    
    return z̄[7] + exp(-model.household.r * T) * ū
end