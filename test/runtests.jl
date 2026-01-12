using Test

@testset "MLPROP.jl" begin
    include("mlprop.jl")
end

@testset "ChemBERTa.jl" begin
    include("chemberta.jl")
end