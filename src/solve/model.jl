abstract type AbstractModelParameters end

struct ModelParameters{H <: Household, F, G, C} <: AbstractModelParameters
    household::H
    firm::F
    government::G
    climate::C
end

"Model parameters specialised to linear labour supply, corresponding to `ϕ = 1`."
struct LinearLabourModelParameters{H <: LinearHousehold, F, G, C} <: AbstractModelParameters
    household::H
    firm::F
    government::G
    climate::C
end

function ModelParameters(household::H, firm::F, government::G, climate::C) where {H, F, G, C}
    isa(household, LinearHousehold) ? 
        LinearLabourModelParameters{H, F, G, C}(household, firm, government, climate) :
        ModelParameters{H, F, G, C}(household, firm, government, climate)
end