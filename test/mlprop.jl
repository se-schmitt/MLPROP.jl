using MLThermoProperties, Clapeyron, PythonCall, ChemBERTa, JLD2, EntropyScaling

@testset "Models" begin
    include("test_ese.jl")
    include("test_grappa.jl")
    include("test_hanna.jl")
end