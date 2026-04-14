function optimalinvestment!(firmvalue::ValueFunction{3}, statesignalgrid::G, firm) where {G <: Grid{3}}
    abatementspace, logitspace, signalspace = statesignalgrid.axes

    for (i, a) in enumerate(abatementspace)
        for (j, z) in enumerate(logitspace)
            for (k, s) in enumerate(signalspace)
                
                optimize(φ -> begin
                    a′ = f(φ, a, firm) 
                    

                end, (0., 1.))

            end
        end
    end

end