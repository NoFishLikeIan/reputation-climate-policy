function interiorindex(i, j, nφ)
    i + (j - 1) * nφ
end

function ξ(τ, τᶜ, signal::Signal)
    signal.ϵ * (τᶜ - τ) / signal.σ
end

function hamiltonian(τ, aᵢ, φ, m, τᶜ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal::Signal, climate::Climate, government::Government, firm::Firm)
    beliefterm = -φ * ∂ᵩu + φ * (1 - φ) * ∂ᵩ²u / 2

    return (
        government.r * w(m, τ, aᵢ, climate, government, firm)
        + e(aᵢ, firm) * ∂ₘu
        + φ * (1 - φ) * ξ(τ, τᶜ, signal)^2 * beliefterm
    )
end

function interiorderivatives(u, φgrid, mgrid, i, j)
    Δφ = step(φgrid)
    Δm = step(mgrid)

    ∂ₘu = (u[i, j + 1] - u[i, j]) / Δm
    ∂ᵩu = (u[i + 1, j] - u[i - 1, j]) / (2Δφ)
    ∂ᵩ²u = (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) / Δφ^2

    return ∂ₘu, ∂ᵩu, ∂ᵩ²u
end

function optimalinteriortax(aᵢ, φ, m, τᶜⱼ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal::Signal{T}, climate::Climate{T}, government::Government{T}, firm::Firm{T}) where T
    τmax = clamp(τᶜⱼ, zero(T), firm.ν * firm.e₀)

    if τmax ≤ 0
        return zero(T)
    end

    obj = @closure τ -> hamiltonian(τ, aᵢ, φ, m, τᶜⱼ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal, climate, government, firm)
    result = Optim.optimize(obj, zero(T), τmax, Optim.Brent())

    return Optim.minimizer(result)
end

function equilibriuminteriortax(φ, m, τᶜⱼ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal::Signal{T}, climate::Climate{T}, government::Government{T}, firm::Firm{T}) where T
    τmax = clamp(τᶜⱼ, zero(T), firm.ν * firm.e₀)

    if τmax ≤ 0
        return zero(T)
    end

    residual = @closure τ -> optimalinteriortax(aᵇ(τ, φ, τᶜⱼ, firm), φ, m, τᶜⱼ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal, climate, government, firm) - τ
    low = zero(T)
    high = τmax
    lowresidual = residual(low)
    highresidual = residual(high)

    if lowresidual ≤ 0
        return low
    elseif highresidual ≥ 0
        return high
    else
        result = Optim.optimize(τ -> residual(τ)^2, low, high, Optim.Brent())
        return Optim.minimizer(result)
    end
end

function interiorpolicycache(::Type{T}, n) where T
    return (
        τ = Vector{T}(undef, n),
        φ = Vector{T}(undef, n),
        m = Vector{T}(undef, n),
        τᶜ = Vector{T}(undef, n),
        ∂ₘu = Vector{T}(undef, n),
        ∂ᵩu = Vector{T}(undef, n),
        ∂ᵩ²u = Vector{T}(undef, n),
    )
end

function fillinteriorpolicycache!(cache, policy, u::AbstractMatrix{T}, φgrid, mgrid, τᶜ, firm::Firm) where T
    nφ, nm = size(u)
    Δφ = step(φgrid)
    Δm = step(mgrid)
    k = 0

    @inbounds for j in 1:(nm - 1)
        m = mgrid[j]
        τᶜⱼ = τᶜ(m)
        upperτ = clamp(τᶜⱼ, zero(T), firm.ν * firm.e₀)

        for i in 2:(nφ - 1)
            k += 1
            φ = φgrid[i]

            cache.τ[k] = clamp(policy[i, j], zero(T), upperτ)
            cache.φ[k] = φ
            cache.m[k] = m
            cache.τᶜ[k] = τᶜⱼ
            cache.∂ₘu[k] = (u[i, j + 1] - u[i, j]) / Δm
            cache.∂ᵩu[k] = (u[i + 1, j] - u[i - 1, j]) / (2Δφ)
            cache.∂ᵩ²u[k] = (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) / Δφ^2
        end
    end

    return cache
end

function copypolicycache!(policy, cache)
    nφ, nm = size(policy)
    k = 0

    @inbounds for j in 1:(nm - 1)
        policy[1, j] = zero(eltype(policy))
        policy[nφ, j] = zero(eltype(policy))

        for i in 2:(nφ - 1)
            k += 1
            policy[i, j] = cache.τ[k]
        end
    end

    @inbounds for i in 1:nφ
        policy[i, nm] = policy[i, nm - 1]
    end

    return policy
end

function updateinteriorpolicy!(policy, u::AbstractMatrix{T}, φgrid, mgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm; cache = interiorpolicycache(T, (size(u, 1) - 2) * (size(u, 2) - 1)), brenttol = sqrt(eps(T)), brentmaxiters = 100, brentcheckevery = 10) where T
    fillinteriorpolicycache!(cache, policy, u, φgrid, mgrid, τᶜ, firm)

    @inbounds for k in eachindex(cache.τ)
        cache.τ[k] = equilibriuminteriortax(
            cache.φ[k],
            cache.m[k],
            cache.τᶜ[k],
            cache.∂ₘu[k],
            cache.∂ᵩu[k],
            cache.∂ᵩ²u[k],
            signal,
            climate,
            government,
            firm,
        )
    end

    copypolicycache!(policy, cache)

    return policy
end

function interiorfarvalue(φgrid, u̲grid, ūgrid)
    [(1 - φ) * u̲grid[end] + φ * ūgrid[end] for φ in φgrid]
end

function initialinteriorvalue(φgrid, mgrid, u̲grid, ūgrid)
    u = Matrix{eltype(u̲grid)}(undef, length(φgrid), length(mgrid))

    @inbounds for j in eachindex(mgrid)
        for i in eachindex(φgrid)
            φ = φgrid[i]
            u[i, j] = (1 - φ) * u̲grid[j] + φ * ūgrid[j]
        end
    end

    return u
end

function pushinteriormatrix!(I, J, V, i, j, v)
    push!(I, i)
    push!(J, j)
    push!(V, v)
end

function buildinteriorsystem(policy, u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm) where {T, TU <: AbstractArray{T}}
    nφ, nm = size(u)
    n = nφ * nm
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, 6n)
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    rhs = Vector{T}(undef, n)
    Δφ = step(φgrid)
    Δm = step(mgrid)

    @inbounds for j in 1:nm
        for i in 1:nφ
            row = interiorindex(i, j, nφ)

            if i == 1
                pushinteriormatrix!(I, J, V, row, row, one(T))
                rhs[row] = u̲grid[j]
                continue
            elseif i == nφ
                pushinteriormatrix!(I, J, V, row, row, one(T))
                rhs[row] = ūgrid[j]
                continue
            elseif j == nm
                pushinteriormatrix!(I, J, V, row, row, one(T))
                φ = φgrid[i]
                rhs[row] = (1 - φ) * u̲grid[end] + φ * ūgrid[end]
                continue
            end

            φ = φgrid[i]
            m = mgrid[j]
            τ = policy[i, j]
            τᶜⱼ = τᶜ(m)
            aᵢ = aᵇ(τ, φ, τᶜⱼ, firm)
            ξᵢ = ξ(τ, τᶜⱼ, signal)
            driftm = e(aᵢ, firm)
            driftφ = -φ^2 * (1 - φ) * ξᵢ^2
            diffφ = φ^2 * (1 - φ)^2 * ξᵢ^2 / 2

            diagonal = zero(T)

            if driftm > 0
                rate = driftm / Δm
                diagonal -= rate
                pushinteriormatrix!(I, J, V, row, interiorindex(i, j + 1, nφ), -rate)
            end

            if diffφ > 0
                rate = diffφ / Δφ^2
                diagonal -= 2rate
                pushinteriormatrix!(I, J, V, row, interiorindex(i - 1, j, nφ), -rate)
                pushinteriormatrix!(I, J, V, row, interiorindex(i + 1, j, nφ), -rate)
            end

            if driftφ > 0
                rate = driftφ / Δφ
                diagonal -= rate
                pushinteriormatrix!(I, J, V, row, interiorindex(i + 1, j, nφ), -rate)
            elseif driftφ < 0
                rate = -driftφ / Δφ
                diagonal -= rate
                pushinteriormatrix!(I, J, V, row, interiorindex(i - 1, j, nφ), -rate)
            end

            pushinteriormatrix!(I, J, V, row, row, government.r + Δt⁻¹ - diagonal)

            rhs[row] = government.r * w(m, τ, aᵢ, climate, government, firm) + Δt⁻¹ * u[i, j]
        end
    end

    return SA.sparse(I, J, V, n, n), rhs
end

function interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm)
    A, rhs = buildinteriorsystem(policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)
    solution = A \ rhs

    @inbounds for k in eachindex(solution)
        nextu[k] = solution[k]
    end

    return nextu
end

function interiorerrors(nextu::TU, u::TU) where {T, TU <: AbstractMatrix{T}}
    abserror = zero(T)
    relerror = zero(T)
    ϵ = eps(T)

    @inbounds for k in eachindex(u)
        error = nextu[k] - u[k]
        abserror = max(abserror, abs(error))
        relerror = max(relerror, abs(error) / max(abs(u[k]), ϵ))
    end

    return abserror, relerror
end

function solveinteriorhjb!(u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm; maxiters = 1000, abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹ = 100., brenttol = sqrt(eps(T)), brentmaxiters = 100, brentcheckevery = 10) where {T, TU <: AbstractMatrix{T}}
    policy = zeros(T, size(u))
    nextu = copy(u)
    policycache = interiorpolicycache(T, (size(u, 1) - 2) * (size(u, 2) - 1))
    abserror = T(Inf)
    relerror = T(Inf)

    for iter in 1:maxiters
        updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm; cache = policycache, brenttol = brenttol, brentmaxiters = brentmaxiters, brentcheckevery = brentcheckevery)
        interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)

        abserror, relerror = interiorerrors(nextu, u)

        u .= nextu

        if verbose > 0
            @printf "Interior iteration %d, errors: abs = %.4e, rel = %.4e\r" iter abserror relerror
        end

        if abserror < abstol && relerror < reltol
            updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm; cache = policycache, brenttol = brenttol, brentmaxiters = brentmaxiters, brentcheckevery = brentcheckevery)
            return u, policy, (iter, abserror, relerror)
        end
    end

    if verbose > 0
        @warn @sprintf "Interior convergence failed after %d iterations with errors: abs = %.4e (%.4e), rel = %.4e (%.4e)" maxiters abserror abstol relerror reltol
    end

    updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm; cache = policycache, brenttol = brenttol, brentmaxiters = brentmaxiters, brentcheckevery = brentcheckevery)
    return u, policy, (maxiters, abserror, relerror)
end
