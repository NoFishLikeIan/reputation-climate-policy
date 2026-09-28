abstract type AbstractModelParameters end

struct ModelParameters{H <: Household, F, C} <: AbstractModelParameters
    household::H
    firm::F
    climate::C
end

"Model parameters specialised to linear labour supply, corresponding to `ϕ = 1`."
struct LinearLabourModelParameters{H <: LinearHousehold, F, C} <: AbstractModelParameters
    household::H
    firm::F
    climate::C
end

function ModelParameters(household::H, firm::F, climate::C) where {H, F, C}
    isa(household, LinearHousehold) ? 
        LinearLabourModelParameters{H, F, C}(household, firm, climate) :
        ModelParameters{H, F, C}(household, firm, climate)
end