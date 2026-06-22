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
    d²φ = φ^2 * (1 - φ)^2 * ξᵢ^2 / 2

    v = government.r * w(m, τ, aᵢ, climate, government, firm)

    if dm > 0
        v += dm * (u[i, j + 1] - u[i, j]) / Δm
    end

    if d²φ > 0
        v += d²φ * (u[i - 1, j] - 2u[i, j] + u[i + 1, j]) / Δφgrid^2
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
    result = Optim.optimize(obj, 0, maxτ, brent)

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

function initialinteriorvalue(φgrid, mgrid, u̲grid::TU, ūgrid::TU) where {T, TU <: AbstractVector{T}}
    u = Matrix{T}(undef, length(φgrid), length(mgrid))

    @inbounds for j in eachindex(mgrid), i in eachindex(φgrid)
        φ = φgrid[i]
        u[i, j] = (1 - φ) * u̲grid[j] + φ * ūgrid[j]
    end

    return u
end

function buildinteriorsystem(policy, u::TU, φgrid, mgrid, u̲grid, ūgrid, τᶜ, Δt⁻¹, signal::Signal, climate::Climate, government::Government, firm::Firm) where {T, TU <: AbstractArray{T}}
    nφ, nm = size(u)
    n = nφ * nm

    I = Int[]; sizehint!(I, 5n)
    J = Int[]; sizehint!(J, 5n)
    V = T[]; sizehint!(V, 5n)

    rhs = similar(vec(u))
    Δφgrid = step(φgrid)
    Δm = step(mgrid)

    @inbounds for j in 1:nm
        for i in 1:nφ
            row = interiorindex(i, j, nφ)

            if i == 1
                pushatstencil!((I, J, V), (row, row), one(T))
                rhs[row] = u̲grid[j]
                continue
            elseif i == nφ
                pushatstencil!((I, J, V), (row, row), one(T))
                rhs[row] = ūgrid[j]
                continue
            elseif j == nm
                pushatstencil!((I, J, V), (row, row), one(T))
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
            d²φ = φ^2 * (1 - φ)^2 * ξᵢ^2 / 2

            diagonal = government.r + Δt⁻¹

            if dm > 0
                rate = dm / Δm
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i, j + 1, nφ)), -rate)
            end

            if d²φ > 0
                rate = d²φ / Δφgrid^2
                diagonal += 2rate
                pushatstencil!((I, J, V), (row, interiorindex(i - 1, j, nφ)), -rate)
                pushatstencil!((I, J, V), (row, interiorindex(i + 1, j, nφ)), -rate)
            end

            if dφ > 0
                rate = dφ / Δφgrid
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i + 1, j, nφ)), -rate)
            elseif dφ < 0
                rate = -dφ / Δφgrid
                diagonal += rate
                pushatstencil!((I, J, V), (row, interiorindex(i - 1, j, nφ)), -rate)
            end

            pushatstencil!((I, J, V), (row, row), diagonal)

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
