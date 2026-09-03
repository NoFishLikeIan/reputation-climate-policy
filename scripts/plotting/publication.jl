const publicationtheme = CairoMakie.Theme(
    fontsize = 16,
    Axis = (;
        titlesize = 18,
        titlegap = 8,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xgridcolor = (:black, 0.08),
        ygridcolor = (:black, 0.08),
        topspinevisible = false,
        rightspinevisible = false,
    ),
    Legend = (;
        labelsize = 13,
        framevisible = false,
    ),
    publication = (;
        samplepathlinewidth = 1.0,
        medianlinewidth = 3.5,
        committedlinewidth = 3.0,
        guidelinewidth = 2.0,
        samplepathopacity = 0.14,
        intervalopacity = 0.22,
        paneltitlefontsize = 20,
        annotationfontsize = 13,
        panelwidth = 300,
        panelheight = 320,
    ),
)

function publicationdefault(key::Symbol)
    CairoMakie.to_value(publicationtheme[:publication, key])
end

function savepublicationfigure(basename, figure)
    CairoMakie.save("$basename.pdf", figure; pt_per_unit = 1)
    CairoMakie.save("$basename.png", figure; px_per_unit = 2)
end
