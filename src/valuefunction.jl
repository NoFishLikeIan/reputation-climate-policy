struct ValueFunction{N, T, TV <: AbstractArray{T, N}, TP <: AbstractArray{T, N}}
    V::TV
    P::TP
end

function ValueFunction(N, T, dims)
    V = Array{T, N}(undef, dims)
    P = similar(V)
    return ValueFunction(V, P)
end

function ValueFunction(grid::G) where {N, T, G <: AbstractGrid{N, T}}
    ValueFunction(N, T, size(grid))
end

function ValueFunction(grid::G, signal::Signal) where {N, T, G <: AbstractGrid{N, T}}
    ValueFunction(N + 1, T, (size(grid)..., length(signal.space[1])))
end

function Base.copy(valuefunction::V) where V <: ValueFunction
    V(copy(valuefunction.V), copy(valuefunction.P))
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: ValueFunction
    copyto!(tovalue.V, fromvalue.V)
    copyto!(tovalue.P, fromvalue.P)
end