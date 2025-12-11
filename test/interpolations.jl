using Test

include("../src/utils.jl")

@testset "interpolate 1D" begin
    # f(x) = 2x + 1 on grid
    xs = range(0.0, 1.0; length = 6)
    vals = @. 2xs + 1

    @test interpolate(vals, 0.0, xs) ≈ 1.0              # left edge
    @test interpolate(vals, 1.0, xs) ≈ 3.0              # right edge
    @test interpolate(vals, -0.1, xs) ≈ 1.0             # below grid clamps
    @test interpolate(vals, 1.1, xs) ≈ 3.0              # above grid clamps
    @test interpolate(vals, 0.25, xs) ≈ (2*0.25 + 1)    # interior linear
    @test interpolate(vals, 0.55, xs) ≈ (2*0.55 + 1)
end

@testset "interpolate 2D bilinear" begin
    # f(x,y) = x + 2y on a regular grid
    xs = range(0.0, 1.0; length = 5)
    ys = range(-1.0, 1.0; length = 5)
    V = [x + 2y for x in xs, y in ys]

    # inside grid
    @test interpolate(V, (0.5, 0.0), (xs, ys)) ≈ (0.5 + 0.0*2)
    @test interpolate(V, (0.2, -0.6), (xs, ys)) ≈ (0.2 + 2*(-0.6)) atol = 1e-12
    @test interpolate(V, (0.9, 0.8), (xs, ys)) ≈ (0.9 + 2*0.8) atol = 1e-12

    # outside y clamps to edge, then 1D in x
    @test interpolate(V, (0.3, -2.0), (xs, ys)) ≈ (0.3 + 2*ys[1])
    @test interpolate(V, (0.7, 3.0), (xs, ys)) ≈ (0.7 + 2*ys[end])

    # outside x handled by inner 1D interpolate edges
    @test interpolate(V, (-0.5, 0.0), (xs, ys)) ≈ (xs[1] + 2*0.0)
    @test interpolate(V, (2.0, 0.0), (xs, ys)) ≈ (xs[end] + 2*0.0)
end