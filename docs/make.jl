using Documenter
using DocumenterVitepress
using DocumenterCitations
using MLPROP, Clapeyron

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    sitename = "MLPROP.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/se-schmitt/MLPROP.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Tutorials" => "tutorials.md",
        "Models" => "models.md",
        "References" => "references.md"
    ],
    plugins=[bib]
)

# DocumenterVitepress.deploydocs(;
#     repo = "github.com/se-schmitt/MLPROP.jl",
#     target = joinpath(@__DIR__, "build"),
#     branch = "gh-pages",
#     devbranch = "main",
#     push_preview = true,
# )