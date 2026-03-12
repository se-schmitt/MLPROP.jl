include("utils.jl")

include("ese.jl")
include("grappa.jl")
include("HANNA/hanna.jl")

# Show method
const MLPROP_MODELS = Union{GRAPPA, ogHANNA, multHANNA, ESE}
function Base.show(io::IO, ::MIME"text/plain", model::MLPROP_MODELS)
    print(io, nameof(typeof(model)))
    length(model) == 1 && println(io, " with 1 component:")
    length(model) > 1 && println(io, " with ", length(model), " components:")
    CL.show_pairs(io,CL.component_list(model))
    
    CL.show_info(io,model)
    CL.show_params(io,model)
    CL.show_reference_state(io,model)
    CL.may_show_references(io,model)
end
CL.show_reference_state(io,::ESE) = nothing

# placeholder function for RDKit #TODO move to utils??
function _get_descriptors_error(smiles)
    error("To use this functionality in `MLPROP.jl`, you need to install and import either `PythonCall.jl` or `RDKitMinimalLib.jl`!")
    return nothing
end

const _get_descriptors = Ref{Function}(_get_descriptors_error)

function get_descriptors(smiles)
    return _get_descriptors[](smiles)
end