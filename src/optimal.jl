function investment(τ, ∂vₐ, firm::Firm)
    @unpack κ, ν, β = firm

    return (β * (τ - ∂vₐ) - κ) / ν
end