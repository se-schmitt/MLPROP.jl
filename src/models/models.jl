include("utils.jl")

include("ese.jl")
include("grappa.jl")
include("HANNA/hanna.jl")

# Show method
const MLPROP_MODELS = Union{GRAPPA, ogHANNA, multHANNA}
function Base.show(io::IO, mime::MIME"text/plain", model::MLPROP_MODELS)
    print(io, nameof(typeof(model)))
    length(model) == 1 && println(io, " with 1 component:")
    length(model) > 1 && println(io, " with ", length(model), " components:")
    CL.show_pairs(io,CL.component_list(model))
    
    CL.show_info(io,model)
    CL.show_params(io,model)
    CL.show_reference_state(io,model)
    CL.may_show_references(io,model)
end

# placeholder function for RDKit #TODO move to utils??
function get_descriptors(smiles)
    error("""
    To use this functionality in `MLPROP.jl`, you need to install and import either `PythonCall.jl` or `RDKitMinimalLib.jl`!
    """)
end