using Documenter
using FermiSea

const DOCS_ROOT = @__DIR__
const SRC_DIR = joinpath(DOCS_ROOT, "src")
const BUILD_DIR = joinpath(DOCS_ROOT, "build")

if isdir(BUILD_DIR)
    rm(BUILD_DIR; recursive = true, force = true)
end

# Tutorial markdown is checked in. Regenerate it manually when an example source
# changes; normal docs builds should not run simulations.

makedocs(;
    sitename="FermiSea.jl",
    authors="Jack H. Farrell",
    repo=Documenter.Remotes.GitHub("jackhfarrell", "FermiSea.jl"),
    format=Documenter.HTML(; edit_link=nothing,
                           repolink="https://github.com/jackhfarrell/FermiSea.jl",
                           size_threshold=20_000_000,
                           size_threshold_warn=200_000),
    pages=[
        "FermiSea.jl" => "index.md",
        "Tutorials" => [
            "Tutorials" => "tutorials/index.md",
            "1\\. Installation" => "tutorials/installation.md",
            "2\\. Running a simulation" => "tutorials/square_bells.md",
            "3\\. Generating a mesh" => "tutorials/generating_a_mesh.md",
        ],
        "Reference" => [
            "Overview" => "reference/index.md",
            "IsotropicFermiHarmonics2D" => "reference/isotropic_fermi_harmonics_2d.md",
            "Analysis" => "reference/analysis.md",
            "Output and Logging" => "reference/output.md",
        ],
    ],
)
