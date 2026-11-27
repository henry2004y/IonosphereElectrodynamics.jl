using Documenter
using DocumenterVitepress
using IonosphereElectrodynamics

include(joinpath(@__DIR__, "..", "examples", "fleks_analytic_polar.jl"))

run_fleks_analytic_polar(;
    outpath = joinpath(@__DIR__, "src", "assets", "fleks_analytic_polar.png"),
)

DocMeta.setdocmeta!(
    IonosphereElectrodynamics,
    :DocTestSetup,
    :(using IonosphereElectrodynamics, StaticArrays);
    recursive = true,
)

makedocs(;
    modules = [IonosphereElectrodynamics],
    authors = "Hongyang Zhou <hyzhou@umich.edu> and contributors",
    repo = Remotes.GitHub("henry2004y", "IonosphereElectrodynamics.jl"),
    sitename = "IonosphereElectrodynamics.jl",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/henry2004y/IonosphereElectrodynamics.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = get(ENV, "BUILD_VITEPRESS", "true") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting_started.md",
            "Formulation & Physics" => "formulation.md",
            "Coupling & GPU Usage" => "coupling_gpu.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = false,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/henry2004y/IonosphereElectrodynamics.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
