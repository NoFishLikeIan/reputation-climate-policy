struct ValueFunction{N, T, TV <: AbstractArray{T, N}, TP <: AbstractArray{T, N}}
    V::TV
    P::TP
end

struct FirmValue{T, TE <: AbstractMatrix{T}, TC <: ValueFunction{3, T}}
    exante::TE
    continuation::TC
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

function FirmValue(grid::G, space::V) where {T, G <: AbstractGrid{2, T}, V <: AbstractVector{T}}
    exante = Matrix{T}(undef, size(grid))
    continuation = ValueFunction(grid, space)

    return FirmValue(exante, continuation)
end

function Base.similar(valuefunction::V) where V <: ValueFunction
    V(similar(valuefunction.V), similar(valuefunction.P))
end

function Base.similar(firmvalue::V) where V <: FirmValue
    FirmValue(similar(firmvalue.exante), similar(firmvalue.continuation))
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: ValueFunction
    copyto!(tovalue.V, fromvalue.V)
    copyto!(tovalue.P, fromvalue.P)

    return tovalue
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: FirmValue
    copyto!(tovalue.exante, fromvalue.exante)
    copyto!(tovalue.continuation, fromvalue.continuation)

    return tovalue
end


function Base.copy(valuefunction::V) where V <: ValueFunction
    newvaluefunction = similar(valuefunction)
    copyto!(newvaluefunction, valuefunction)
    
    return newvaluefunction
end

function Base.copy(firmvalue::V) where V <: FirmValue
    newfirmvalue = similar(firmvalue)
    copyto!(newfirmvalue, firmvalue)

    return newfirmvalue
end
