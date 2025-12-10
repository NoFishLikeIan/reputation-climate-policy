using Test
using BenchmarkTools

include("../src/utils.jl")

@testset "interpolate 1d float64" begin
    grid = 0.0:0.5:2.0
    vals = [0.0, 1.0, 2.0, 3.0, 4.0] # linear f(x) = 2x

    @test interpolate(vals, 0.25, grid) ≈ 0.5
    @test interpolate(vals, 0.75, grid) ≈ 1.5
    @test interpolate(vals, -1.0, grid) == vals[1]
    @test interpolate(vals, 3.0, grid) == vals[end]
end

@testset "interpolate 1d bigfloat" begin
    grid = big"0.0":big"0.5":big"1.5"
    vals = BigFloat.([0, 1, 2, 3]) # still f(x) = 2x

    @test interpolate(vals, big"0.75", grid) ≈ big"1.5"
    @test interpolate(vals, big"-2.0", grid) == vals[1]
    @test interpolate(vals, big"2.0", grid) == vals[end]
end

@testset "interpolate 2d float32" begin
    xgrid = 0f0:1f0:2f0
    ygrid = 0f0:1f0:2f0
    mat = Float32[x + 2y for y in ygrid, x in xgrid] # f(x,y) = x + 2y

    @test interpolate(mat, (0.25f0, 0.75f0), (xgrid, ygrid)) ≈ Float32(0.25 + 2 * 0.75)
    @test interpolate(mat, (3f0, -3f0), (xgrid, ygrid)) == mat[1, end] # clamps independently
    @test interpolate(mat, (-3f0, 3f0), (xgrid, ygrid)) == mat[end, 1]
end

@testset "interpolate 2d bigfloat" begin
    xgrid = big"0.0":big"0.5":big"1.0"
    ygrid = big"0.0":big"0.5":big"1.0"
    mat = [x + y for y in ygrid, x in xgrid] # f(x,y) = x + y

    @test interpolate(mat, (big"0.25", big"0.75"), (xgrid, ygrid)) ≈ big"1.0"
    @test interpolate(mat, (big"2.0", big"-2.0"), (xgrid, ygrid)) == mat[1, end]
end

@testset "interpolate 2d int" begin
    xgrid = 0:1:2
    ygrid = 0:1:2
    mat = [x + 3y for y in ygrid, x in xgrid] # f(x,y) = x + 3y

    @test interpolate(mat, (1, 1), (xgrid, ygrid)) == 4
    @test interpolate(mat, (1, 0), (xgrid, ygrid)) == 1
    @test interpolate(mat, (1, 2), (xgrid, ygrid)) == 7
    @test interpolate(mat, (-2, 5), (xgrid, ygrid)) == mat[end, 1]
end

# simple benchmark to watch allocations and typical timing
begin
    xgrid = 0f0:1f0:2f0
    ygrid = 0:1:2
    mat = Float32[0 1 2; 2 3 4; 4 5 6]
    xpoint = 0.3
    ypoint = 0.7

    @benchmark interpolate($mat, ($xpoint, $ypoint), ($xgrid, $ygrid))
end