using Documenter
using FermiSea
using Literate

Literate.markdown(joinpath(@__DIR__, "tutorials", "square_bells.jl"),
                  joinpath(@__DIR__, "src", "tutorials");
                  documenter = true, execute = false, credit = false)

makedocs(;
    sitename = "FermiSea.jl",
    authors = "Jack H. Farrell",
    modules = [FermiSea],
    checkdocs = :exports,
    repo = Documenter.Remotes.GitHub("jackhfarrell", "FermiSea.jl"),
    format = Documenter.HTML(;
        prettyurls = false,
        edit_link = "main",
        repolink = "https://github.com/jackhfarrell/FermiSea.jl",
        assets = [
            "assets/custom.css",
            asset("assets/logo.svg?v=2"; class = :ico, islocal = true),
        ],
        footer = "FermiSea.jl · steady transport on a circular Fermi surface",
    ),
    pages = [
        "FermiSea.jl" => "index.md",
        "Model and conventions" => "model.md",
        "Tutorials" => ["Square bells" => "tutorials/square_bells.md"],
        "Solving" => "solving.md",
        "Currents and saved results" => "analysis.md",
        "API reference" => "api.md",
    ],
)
