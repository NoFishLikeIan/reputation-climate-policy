function rightboundaryvalue(m, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    a = aᶜ(τᶜ, firm)
    eᶜ = e(a, firm)
    δm = climate.δₘ
    mᶜ = eᶜ / δm
    Δm = m - mᶜ

    damage = mᶜ^2 + 2 * mᶜ * Δm * government.r / (government.r + δm) + Δm^2 * government.r / (government.r + 2 * δm)

    return c(a, firm) + government.y₀ * climate.ξ * damage / 2
end

function rightboundary(mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    u = [rightboundaryvalue(m, τᶜ, climate, government, firm) for m in mgrid]
    τ = zeros(length(mgrid))
    residual = rightboundaryresidual(u, mgrid, τᶜ, climate, government, firm)

    return (mgrid = mgrid, u = u, τ = τ, residual = residual)
end

function boundaryderivative(u, mgrid, i, q)
    if q > 0 && i < length(mgrid)
        return (u[i + 1] - u[i]) / (mgrid[i + 1] - mgrid[i])
    elseif q < 0 && i > 1
        return (u[i] - u[i - 1]) / (mgrid[i] - mgrid[i - 1])
    else
        return zero(eltype(u))
    end
end

function leftboundaryhamiltonian(τ, u, mgrid, i, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    m = mgrid[i]
    a = aᵇ(τ, 0., τᶜ, firm)
    q = concentrationdrift(m, a, climate, firm)
    p = boundaryderivative(u, mgrid, i, q)

    return government.r * w(m, τ, a, climate, government, firm) + q * p
end

function updateleftpolicy(u, mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm; τbar = firm.ν * firm.e₀)
    τ = similar(u)

    for i in eachindex(mgrid)
        obj = τᵢ -> leftboundaryhamiltonian(τᵢ, u, mgrid, i, τᶜ, climate, government, firm)
        result = Optim.optimize(obj, zero(τbar), τbar, Optim.Brent())
        τ[i] = Optim.minimizer(result)
    end

    return τ
end

function evaluateleftpolicy(τ, mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    n = length(mgrid)
    lower = zeros(n - 1)
    diagonal = fill(government.r, n)
    upper = zeros(n - 1)
    rhs = zeros(n)

    for i in eachindex(mgrid)
        m = mgrid[i]
        a = aᵇ(τ[i], 0., τᶜ, firm)
        q = concentrationdrift(m, a, climate, firm)

        rhs[i] = government.r * w(m, τ[i], a, climate, government, firm)

        if q > 0 && i < n
            Δm = mgrid[i + 1] - mgrid[i]
            diagonal[i] += q / Δm
            upper[i] -= q / Δm
        elseif q < 0 && i > 1
            Δm = mgrid[i] - mgrid[i - 1]
            diagonal[i] -= q / Δm
            lower[i - 1] += q / Δm
        end
    end

    return LinearAlgebra.Tridiagonal(lower, diagonal, upper) \ rhs
end

function leftboundaryresidual(u, τ, mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    residual = zero(eltype(u))

    for i in eachindex(mgrid)
        m = mgrid[i]
        a = aᵇ(τ[i], 0., τᶜ, firm)
        q = concentrationdrift(m, a, climate, firm)
        p = boundaryderivative(u, mgrid, i, q)
        resid = government.r * u[i] - government.r * w(m, τ[i], a, climate, government, firm) - q * p
        residual = max(residual, abs(resid))
    end

    return residual
end

function rightboundaryresidual(u, mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm)
    residual = zero(eltype(u))
    a = aᶜ(τᶜ, firm)

    for i in eachindex(mgrid)
        m = mgrid[i]
        q = concentrationdrift(m, a, climate, firm)
        p = boundaryderivative(u, mgrid, i, q)
        resid = government.r * u[i] - government.r * w(m, zero(τᶜ), a, climate, government, firm) - q * p
        residual = max(residual, abs(resid))
    end

    return residual
end

function solveleftboundary(mgrid, τᶜ, climate::Climate, government::Government, firm::StaticFirm; tolerance = 1e-8, maxiter = 500, verbose = false, τbar = firm.ν * firm.e₀)
    τ = zeros(length(mgrid))
    u = evaluateleftpolicy(τ, mgrid, τᶜ, climate, government, firm)
    residual = leftboundaryresidual(u, τ, mgrid, τᶜ, climate, government, firm)
    converged = false
    iteration = 0

    for iter in 1:maxiter
        iteration = iter
        oldτ = copy(τ)
        τ = updateleftpolicy(u, mgrid, τᶜ, climate, government, firm; τbar)
        u = evaluateleftpolicy(τ, mgrid, τᶜ, climate, government, firm)
        residual = leftboundaryresidual(u, τ, mgrid, τᶜ, climate, government, firm)
        change = maximum(abs.(τ .- oldτ))

        if verbose
            @printf "Boundary iteration %d: policy change %.3e, residual %.3e\n" iter change residual
        end

        if change < tolerance
            converged = true
            break
        end
    end

    return (mgrid = mgrid, u = u, τ = τ, residual = residual, iterations = iteration, converged = converged)
end
