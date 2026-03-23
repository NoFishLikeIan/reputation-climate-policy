struct Value{NS, NP, TS, TP, SV <: AbstractArray{NS, TS}, PV <: AbstractArray{NP, TP}}
    value::SV
    policy::PV
end

function Base.copy(V::TV) where TV <: Value
    TV(copy(V.value), copy(V.policy))
end

function update!(to::Value, from::Value)
    to.value .= from.value
    to.policy .= from.policy
end

function optimalinvestment(τ, a, Vⁿ::Value, Sⁿ::Value, A, T, firm::Firm)
    objective = φ -> begin
        a′ = f(φ, a, firm)
        τ′ = interpolate(Sⁿ.policy, a′, A)
        return c(φ, firm) - firm.β * τ * a′ + firm.β * interpolate(Vⁿ.value, (a′, τ′), (A, T))
    end

    sol = optimize(objective, 0, firm.e₀)

    costs = Optim.minimum(sol)
    φ = Optim.minimizer(sol)

    return costs, φ
end

function optimaltax(a, Sⁿ::Value, Vⁿ::Value, A, T, firm::Firm, government::Government)
    objective = τ -> begin
        φ = interpolate(Vⁿ.policy, (a, τ), (A, T))
        a′ = f(φ, a, firm)
        return c(φ, firm) + d(e(a′, firm), government) + government.β * interpolate(Sⁿ.value, a′, A)
    end

    sol = optimize(objective, T[1], T[end])

    welfare = Optim.minimum(sol)
    τ = Optim.minimizer(sol)

    return welfare, τ 
end

function firmstep!(Vⁿ⁺¹, Vⁿ, Sⁿ, A, T, firm::Firm; ρ = 0.5)
    @inbounds for i in eachindex(A), j in eachindex(T)
        a = A[i]
        τ = T[j]

        costs, φ = optimalinvestment(τ, a, Vⁿ, Sⁿ, A, T, firm)

        Vⁿ⁺¹.value[i, j]  = ρ * costs + (1 - ρ) * Vⁿ.value[i, j]
        Vⁿ⁺¹.policy[i, j] = ρ * φ     + (1 - ρ) * Vⁿ.policy[i, j]
    end
end

function governmentstep!(Sⁿ⁺¹, Sⁿ, Vⁿ, A, T, firm::Firm, government::Government; ρ = 0.5)

    @inbounds for i in eachindex(A)
        a = A[i]

        welfare, τ = optimaltax(a, Sⁿ⁺¹, Vⁿ, A, T, firm, government)

        Sⁿ⁺¹.value[i]  = ρ * welfare + (1 - ρ) * Sⁿ.value[i]
        Sⁿ⁺¹.policy[i] = ρ * τ       + (1 - ρ) * Sⁿ.policy[i]
    end
end

function steadystate!(Sⁿ, Vⁿ, A, T, firm::Firm, government::Government; ρ = 0.9, totaliter = 10_000, tol = 1e-6, verbose = 0)

    Sⁿ⁺¹ = copy(Sⁿ)
    Vⁿ⁺¹ = copy(Vⁿ)

    for iter in 1:totaliter

        firmstep!(Vⁿ⁺¹, Vⁿ, Sⁿ, A, T, firm; ρ)
        governmentstep!(Sⁿ⁺¹, Sⁿ, Vⁿ, A, T, firm, government; ρ)

        err = max(maximum(abs, Vⁿ⁺¹.value .- Vⁿ.value), maximum(abs, Sⁿ⁺¹.value .- Sⁿ.value))

        if verbose > 0
            @printf "Iteration %d / %d \t error = %.2e\r" iter totaliter err
        end

        if err < tol
            return Sⁿ⁺¹, Vⁿ⁺¹
        end

        update!(Sⁿ, Sⁿ⁺¹)
        update!(Vⁿ, Vⁿ⁺¹)
    end
end