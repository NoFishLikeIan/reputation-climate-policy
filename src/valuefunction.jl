struct ValueFunction{N, T, TV <: AbstractArray{T, N}, TP <: AbstractArray{T, N}}
    V::TV
    P::TP
end

function ValueFunction(N, T, dims)
    V = Array{T, N}(undef, dims)
    P = similar(V)
    return ValueFunction(V, P)
end

function ValueFunction(grid::V) where {T, V <: AbstractVector{T}}
    ValueFunction(1, T, (length(grid), ))
end

function ValueFunction(grid::G) where {N, T, G <: AbstractGrid{N, T}}
    ValueFunction(N, T, size(grid))
end

function ValueFunction(grid::G, space::V) where {N, T, G <: AbstractGrid{N, T}, V <: AbstractVector{T}}
    ValueFunction(N + 1, T, (size(grid)..., length(space)))
end

function Base.similar(valuefunction::V) where V <: ValueFunction
    V(similar(valuefunction.V), similar(valuefunction.P))
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: ValueFunction
    copyto!(tovalue.V, fromvalue.V)
    copyto!(tovalue.P, fromvalue.P)
end


function Base.copy(valuefunction::V) where V <: ValueFunction
    newvaluefunction = similar(valuefunction)
    copyto!(newvaluefunction, valuefunction)
    
    return newvaluefunction
end
