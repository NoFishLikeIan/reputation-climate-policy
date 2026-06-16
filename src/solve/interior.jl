function interiorindex(i, j, nφ)
    i + (j - 1) * nφ
end

function ξ(τ, τᶜ, signal::Signal)
    signal.ϵ * (τᶜ - τ) / signal.σ
end

function interiorhamiltonian(τ, φ, m, τᶜ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal::Signal, climate::Climate, government::Government, firm::Firm)
    aᵢ = aᵇ(τ, φ, τᶜ, firm)
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

function optimalinteriortax(φ, m, τᶜ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal::Signal, climate::Climate, government::Government, firm::Firm)
    upperτ = clamp(τᶜ, zero(τᶜ), firm.ν * firm.e₀)

    if upperτ ≤ 0
        return zero(upperτ)
    end

    obj = τ -> interiorhamiltonian(τ, φ, m, τᶜ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal, climate, government, firm)
    result = Optim.optimize(obj, zero(upperτ), upperτ, Optim.Brent())

    return Optim.minimizer(result)
end

function updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    nφ, nm = size(u)

    @inbounds for j in 1:(nm - 1)
        m = mgrid[j]
        τᶜⱼ = τᶜ(m)

        policy[1, j] = zero(τᶜⱼ)
        policy[nφ, j] = zero(τᶜⱼ)

        for i in 2:(nφ - 1)
            φ = φgrid[i]
            ∂ₘu, ∂ᵩu, ∂ᵩ²u = interiorderivatives(u, φgrid, mgrid, i, j)

            policy[i, j] = optimalinteriortax(φ, m, τᶜⱼ, ∂ₘu, ∂ᵩu, ∂ᵩ²u, signal, climate, government, firm)
        end
    end

    policy[:, nm] .= policy[:, nm - 1]

    return policy
end

function interiorfarvalue(φgrid, u̲, ū)
    [(1 - φ) * u̲[end] + φ * ū[end] for φ in φgrid]
end

function initialinteriorvalue(φgrid, mgrid, u̲, ū)
    u = Matrix{eltype(u̲)}(undef, length(φgrid), length(mgrid))

    @inbounds for j in eachindex(mgrid)
        for i in eachindex(φgrid)
            φ = φgrid[i]
            u[i, j] = (1 - φ) * u̲[j] + φ * ū[j]
        end
    end

    return u
end

function pushinteriorgenerator!((I, J, V), i, j, v)
    push!(I, i)
    push!(J, j)
    push!(V, v)
end

function pushinteriormatrix!((I, J, V), i, j, v)
    push!(I, i)
    push!(J, j)
    push!(V, v)
end

function buildinteriorsystem(policy, u, φgrid, mgrid, u̲, ū, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm)
    nφ, nm = size(u)
    n = nφ * nm
    I = Int[]
    J = Int[]
    V = eltype(u)[]
    rhs = similar(vec(u))
    Δφ = step(φgrid)
    Δm = step(mgrid)
    ufar = interiorfarvalue(φgrid, u̲, ū)

    @inbounds for j in 1:nm
        for i in 1:nφ
            row = interiorindex(i, j, nφ)

            if i == 1
                pushinteriormatrix!((I, J, V), row, row, one(eltype(u)))
                rhs[row] = u̲[j]
                continue
            elseif i == nφ
                pushinteriormatrix!((I, J, V), row, row, one(eltype(u)))
                rhs[row] = ū[j]
                continue
            elseif j == nm
                pushinteriormatrix!((I, J, V), row, row, one(eltype(u)))
                rhs[row] = ufar[i]
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

            generator = (Int[], Int[], eltype(u)[])
            diagonal = zero(eltype(u))

            if driftm > 0
                rate = driftm / Δm
                diagonal -= rate
                pushinteriorgenerator!(generator, row, interiorindex(i, j + 1, nφ), rate)
            end

            if diffφ > 0
                rate = diffφ / Δφ^2
                diagonal -= 2rate
                pushinteriorgenerator!(generator, row, interiorindex(i - 1, j, nφ), rate)
                pushinteriorgenerator!(generator, row, interiorindex(i + 1, j, nφ), rate)
            end

            if driftφ > 0
                rate = driftφ / Δφ
                diagonal -= rate
                pushinteriorgenerator!(generator, row, interiorindex(i + 1, j, nφ), rate)
            elseif driftφ < 0
                rate = -driftφ / Δφ
                diagonal -= rate
                pushinteriorgenerator!(generator, row, interiorindex(i - 1, j, nφ), rate)
            end

            pushinteriorgenerator!(generator, row, row, diagonal)

            pushinteriormatrix!((I, J, V), row, row, government.r + Δt⁻¹)
            for k in eachindex(generator[1])
                pushinteriormatrix!((I, J, V), generator[1][k], generator[2][k], -generator[3][k])
            end

            rhs[row] = government.r * w(m, τ, aᵢ, climate, government, firm) + Δt⁻¹ * u[i, j]
        end
    end

    return SA.sparse(I, J, V, n, n), rhs
end

function interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲, ū, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm)
    A, rhs = buildinteriorsystem(policy, u, φgrid, mgrid, u̲, ū, τᶜ, Δt⁻¹, signal, climate, government, firm)
    nextu .= reshape(A \ rhs, size(u))

    return nextu
end

function solveinteriorhjb!(u::TU, φgrid, mgrid, u̲, ū, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm; maxiters = 1000, abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹ = 100.) where {T, TU <: AbstractMatrix{T}}
    policy = similar(u)
    nextu = copy(u)
    errors = similar(u)
    abserror = T(Inf)
    relerror = T(Inf)

    for iter in 1:maxiters
        updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
        interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲, ū, τᶜ, Δt⁻¹, signal, climate, government, firm)

        errors .= nextu .- u
        abserror = maximum(abs, errors)
        relerror = maximum(abs.(errors) ./ max.(abs.(u), eps(T)))

        u .= nextu

        if verbose > 0
            @printf "Interior iteration %d, errors: abs = %.4e, rel = %.4e\r" iter abserror relerror
        end

        if abserror < abstol && relerror < reltol
            updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
            return u, policy, (iter, abserror, relerror)
        end
    end

    if verbose > 0
        @warn @sprintf "Interior convergence failed after %d iterations with errors: abs = %.4e (%.4e), rel = %.4e (%.4e)" maxiters abserror abstol relerror reltol
    end

    updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
    return u, policy, (maxiters, abserror, relerror)
end
