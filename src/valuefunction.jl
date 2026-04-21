struct ValueFunction{N, T, TV <: AbstractArray{T, N}, TP <: AbstractArray{T, N}}
    V::TV
    P::TP
end

struct FirmValue{T, TC <: ValueFunction{2, T}, TE <: ValueFunction{3, T}}
    continuation::TC
    expost::TE
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
    continuation = ValueFunction(grid)
    expost = ValueFunction(grid, space)

    return FirmValue(continuation, expost)
end

function Base.similar(valuefunction::V) where V <: ValueFunction
    V(similar(valuefunction.V), similar(valuefunction.P))
end

function Base.similar(firmvalue::V) where V <: FirmValue
    FirmValue(similar(firmvalue.continuation), similar(firmvalue.expost))
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: ValueFunction
    copyto!(tovalue.V, fromvalue.V)
    copyto!(tovalue.P, fromvalue.P)

    return tovalue
end

function Base.copyto!(tovalue::V, fromvalue::V) where V <: FirmValue
    copyto!(tovalue.continuation, fromvalue.continuation)
    copyto!(tovalue.expost, fromvalue.expost)

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
