using Test

@testset "MLThermoProperties.jl" begin
    include("mlprop.jl")
end

@testset "ChemBERTa.jl" begin
    include("chemberta.jl")
end