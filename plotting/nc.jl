function plotvalue(A, value::Value)
    fig = plot()

    return plotvalue!(fig, A, value)
end

function plotvalue!(fig, A, value::Value; plotkwargs...)
    plot!(fig, A, value.state; xlims = extrema(A),  plotkwargs...)
    twinfig = twinx(fig)
    plot!(twinfig, A, value.policy; xlims = extrema(A),  plotkwargs...)
end