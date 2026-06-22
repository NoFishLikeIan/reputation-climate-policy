const brent = Optim.Brent()

function pushatstencil!((I, J, V), (i, j), v)
    push!(I, i)
    push!(J, j)
    push!(V, v)
end

function interiorindex(i, j, nφ)
    i + (j - 1) * nφ
end