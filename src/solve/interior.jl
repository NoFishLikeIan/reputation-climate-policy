function interiorindex(i, j, nφ)
    i + (j - 1) * nφ
end

function ξ(τ, τᶜ, signal::Signal)
    signal.ϵ * (τᶜ - τ) / signal.σ
end

function discretehamiltonian(τ, i, j, u, φgrid, mgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm)
    Δφgrid = step(φgrid)
    Δm = step(mgrid)

    φ = φgrid[i]
    m = mgrid[j]

    aᵢ = aᵇ(τ, φ, τᶜ, firm)
    ξᵢ = ξ(τ, τᶜ, signal)

    dm = e(aᵢ, firm)
    dφ = -φ^2 * (1 - φ) * ξᵢ^2
    diffusionφ = φ^2 * (1 - φ)^2 * ξᵢ^2 / 2

    v = government.r * w(m, τ, aᵢ, climate, government, firm)

    if dm > 0
        v += dm * (u[i, j + 1] - u[i, j]) / Δm
    end

    if diffusionφ > 0
        v += diffusionφ * (u[i - 1, j] - 2u[i, j] + u[i + 1, j]) / Δφgrid^2
    end

    if dφ > 0
        v += dφ * (u[i + 1, j] - u[i, j]) / Δφgrid
    elseif dφ < 0
        v += dφ * (u[i, j] - u[i - 1, j]) / Δφgrid
    end

    return v
end

function optimalinteriortax(i, j, u, φgrid, mgrid, τᶜ, signal::Signal{T}, climate::Climate{T}, government::Government{T}, firm::Firm{T}) where T
    
    maxτ = firm.ν * firm.e₀

    obj = @closure τ -> discretehamiltonian(τ, i, j, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
    result = Optim.optimize(obj, zero(T), maxτ, Optim.Brent())

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
            policy[i, j] = optimalinteriortax(i, j, u, φgrid, mgrid, τᶜⱼ, signal, climate, government, firm)
        end
    end

    policy[:, nm] .= policy[:, nm - 1]

    return policy
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

function pushinteriormatrix!((I, J, V), i, j, v)
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
    sizehint!(I, 5n)
    sizehint!(J, 5n)
    sizehint!(V, 5n)

    rhs = similar(vec(u))
    Δφgrid = step(φgrid)
    Δm = step(mgrid)

    @inbounds for j in 1:nm
        for i in 1:nφ
            row = interiorindex(i, j, nφ)

            if i == 1
                pushinteriormatrix!((I, J, V), row, row, one(T))
                rhs[row] = u̲grid[j]
                continue
            elseif i == nφ
                pushinteriormatrix!((I, J, V), row, row, one(T))
                rhs[row] = ūgrid[j]
                continue
            elseif j == nm
                pushinteriormatrix!((I, J, V), row, row, one(T))
                φ = φgrid[i]
                rhs[row] = (one(T) - φ) * u̲grid[end] + φ * ūgrid[end]
                continue
            end

            φ = φgrid[i]
            m = mgrid[j]
            τ = policy[i, j]
            τᶜⱼ = τᶜ(m)
            aᵢ = aᵇ(τ, φ, τᶜⱼ, firm)
            ξᵢ = ξ(τ, τᶜⱼ, signal)
            dm = e(aᵢ, firm)
            dφ = -φ^2 * (1 - φ) * ξᵢ^2
            diffusionφ = φ^2 * (1 - φ)^2 * ξᵢ^2 / 2

            diagonal = government.r + Δt⁻¹

            if dm > 0
                rate = dm / Δm
                diagonal += rate
                pushinteriormatrix!((I, J, V), row, interiorindex(i, j + 1, nφ), -rate)
            end

            if diffusionφ > 0
                rate = diffusionφ / Δφgrid^2
                diagonal += 2rate
                pushinteriormatrix!((I, J, V), row, interiorindex(i - 1, j, nφ), -rate)
                pushinteriormatrix!((I, J, V), row, interiorindex(i + 1, j, nφ), -rate)
            end

            if dφ > 0
                rate = dφ / Δφgrid
                diagonal += rate
                pushinteriormatrix!((I, J, V), row, interiorindex(i + 1, j, nφ), -rate)
            elseif dφ < 0
                rate = -dφ / Δφgrid
                diagonal += rate
                pushinteriormatrix!((I, J, V), row, interiorindex(i - 1, j, nφ), -rate)
            end

            pushinteriormatrix!((I, J, V), row, row, diagonal)

            rhs[row] = government.r * w(m, τ, aᵢ, climate, government, firm) + Δt⁻¹ * u[i, j]
        end
    end

    return SA.sparse(I, J, V, n, n), rhs
end

function interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm)
    A, rhs = buildinteriorsystem(policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)
    nextu .= reshape(A \ rhs, size(u))

    return nextu
end

function solveinteriorhjb!(u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, signal::Signal, climate::Climate, government::Government, firm::Firm; maxiters = 1000, abstol = 1e-2, reltol = 1e-2, verbose = 0, Δt⁻¹ = 100.) where {T, TU <: AbstractMatrix{T}}
    policy = similar(u)
    nextu = copy(u)
    errors = similar(u)
    abserror = T(Inf)
    relerror = T(Inf)

    for iter in 1:maxiters
        updateinteriorpolicy!(policy, u, φgrid, mgrid, τᶜ, signal, climate, government, firm)
        interiorhjbstep!(nextu, policy, u, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal, climate, government, firm)

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
