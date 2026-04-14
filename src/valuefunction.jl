struct ValueFunction{N, T, TV <: AbstractArray{T, N}, TP <: AbstractArray{T, N}}
    V::TV
    P::TP
end

function ValueFunction(grid::G) where {N, T, G <: AbstractGrid{N, T}}
    ValueFunction(
        Array{T, N}(undef, size(grid)),
        Array{T, N}(undef, size(grid))
    )
end

function ValueFunction(grid::G, init::NTuple{2, T}) where {N, T, G <: AbstractGrid{N, T}}
    ValueFunction(
        ones(size(grid)) * init[1],
        ones(size(grid)) * init[2]
    )
end