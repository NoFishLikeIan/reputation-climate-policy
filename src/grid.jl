abstract type AbstractGrid{N, T, A <: NTuple{N, AbstractVector{T}}} end

Base.ndims(::G) where {N, G <: AbstractGrid{N}} = N
Base.length(grid::G) where G <: AbstractGrid = prod(size(grid))
Base.eltype(::G) where {N, T, G <: AbstractGrid{N, T}} = T

function Base.size(grid::G, dim) where G <: AbstractGrid
    length(getindex(grid.nodes, dim))
end
function Base.size(grid::G) where G <: AbstractGrid
    ntuple(dim -> size(grid, dim), ndims(grid))
end

function Base.axes(grid::G, dim) where G <: AbstractGrid
    only(axes(getindex(grid.nodes, dim)))
end
function Base.axes(grid::G) where G <: AbstractGrid
    ntuple(dim -> axes(grid, dim), ndims(grid))
end

function bounds(grid::G, dim) where G <: AbstractGrid
    extrema(getindex(grid.nodes, dim))
end
function bounds(grid::G) where G <: AbstractGrid
    ntuple(dim -> bounds(grid, dim), ndims(grid))
end

struct UniformGrid{N, T, A <: NTuple{N, AbstractRange{T}}} <: AbstractGrid{N, T, A}
	nodes::A
end

function Base.step(grid::UniformGrid, dim)
    step(getindex(grid.nodes, dim))
end
function steps(grid::UniformGrid)
    ntuple(dim -> step(grid, dim), ndims(grid))
end

struct Grid{N, T, A <: NTuple{N, AbstractVector{T}}} <: AbstractGrid{N, T, A}
    nodes::A
end