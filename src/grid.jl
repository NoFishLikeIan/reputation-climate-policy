Domain{T} = NTuple{2, T}

struct Grid{N, T, A <: AbstractRange{T}}
    domains::NTuple{N, Domain{T}}
    ranges::NTuple{N, A}

    function Grid(Ns::NTuple{N, Int}, domains::NTuple{N, Domain{T}}) where {T, N}
        ranges = ntuple(i -> range(domains[i]..., Ns[i]), Val(N))
        new{N, T, typeof(ranges[1])}(domains, ranges)
    end
end
Base.size(grid::Grid, n) = length(grid.ranges[n])
Base.size(grid::Grid{N}) where N = ntuple(i -> size(grid, i), Val(N))

Base.axes(grid::Grid, n) = only(axes(grid.ranges[n]))
Base.axes(grid::Grid{N}) where N = ntuple(i -> axes(grid, i), Val(N))

struct FirmValue{T, TV <: AbstractArray{T, 3}, TP <: AbstractArray{T, 3}}
    value::TV
    investment::TP

    function FirmValue(stategrid::Grid{2, T}, controlgrid::Grid{2, T}) where {T}
        FirmValue(T, stategrid, controlgrid)
    end
    function FirmValue(T, stategrid::Grid{2}, controlgrid::Grid{2})
        V = Array{T}(undef, size(stategrid)..., size(controlgrid, 2))
        P = similar(V)

        return new{T, typeof(V), typeof(P)}(V, P)
    end
    function FirmValue(value::TV, investment::TP) where {T, TV <: AbstractArray{T, 3}, TP <: AbstractArray{T, 3}}
        new{T, TV, TP}(value, investment)
    end
end
function Base.copy(V::FirmValue)
    FirmValue(copy(V.value), copy(V.investment))
end

struct Welfare{T, TS <: AbstractMatrix{T}, TT <: AbstractMatrix{T}}
    welfare::TS
    tax::TT

    function Welfare(stategrid::Grid{2, T}) where {T}
        Welfare(T, stategrid)
    end
    function Welfare(T, stategrid::Grid{2})
        S = Array{T}(undef, size(stategrid))
        Q = similar(S)

        return new{T, typeof(S), typeof(Q)}(S, Q)
    end
    function Welfare(welfare::TS, tax::TT) where {T, TS <: AbstractMatrix{T}, TT <: AbstractMatrix{T}}
        new{T, TS, TT}(welfare, tax)
    end
end
function Base.copy(S::Welfare)
    Welfare(copy(S.welfare), copy(S.tax))
end
