abstract type Grid{N, T} end

struct CommittedGrid{T, TP <: AbstractMatrix, TG <: NTuple{2, <:AbstractVector{T}}} <: Grid{2, T}
    points::TP
    grids::TG
end

function CommittedGrid(order, lb, ub)
    points = FastChebInterp.chebpoints(order, lb, ub)
    agrid = getindex.(points[:, 1], 1)
    mgrid = getindex.(points[1, :], 2)

    return CommittedGrid(points, (agrid, mgrid))
end

Base.size(grid::TG) where TG <: Grid = size(grid.points);
Base.size(grid::TG, dim) where TG <: Grid = size(grid.points, dim)


# Dynamics
function χ(τ, τᶜ, signal::Signal)
    (signal.ϵ / signal.σ) * (τᶜ - τ) 
end

function beliefdrift(χ, φ)
    -φ^2 * (1 - φ) * χ^2
end

function beliefdiffusion(χ, φ)
    φ * (1 - φ) * χ
end