function map_state_dict!(ps, sd; ftype=Float32)
    _sd = OrderedDict(filter(((k,v),) -> !startswith(k,"pooler"), sd))
    ks = String.(keys(_sd))

    # Transformations before split
    replacings = [
        "LayerNorm.weight" => "norm.scale",
        "LayerNorm.bias" => "norm.bias",
    ]
    ks = replace.(ks, replacings...)
    replacings_emb = [
        "embeddings" => "embedding",
        "position_embeddings" => "pos_embed",
        "token_type_embeddings" => "type_embed",
        "word_embeddings" => "word_embed",
    ]
    ks = replace.(ks, replacings_emb...)
    replacings_att = [
        "encoder.layer" => "blocks",
        "attention.output.dense" => "mha.out_proj",
        "attention.output.norm" => "mha_norm",
        "attention.self.key" => "mha.k_proj", 
        "attention.self.query" => "mha.q_proj", 
        "attention.self.value" => "mha.v_proj", 
    ]
    ks = replace.(ks, replacings_att...)
    replacings_mlp = [
        "intermediate.dense" => "mlp.layer_1",
        "output.dense" => "mlp.layer_2",
        "output.norm" => "mlp_norm",
    ]
    ks = replace.(ks, replacings_mlp...)
    # paths = create_path.(ks)

    for ((k_py,v), k) in zip(_sd, ks)
        if v isa AbstractArray
            P = ftype.(v)
            if endswith(k, "weight") && occursin("embed", k) && ndims(P) == 2
                P = permutedims(P, (2, 1))
            end
        else
            P = ftype.(first(v) isa AbstractArray ? hcat(v...)' : vcat(v...))
        end
        pth = create_path(k)
        setpath!(ps, pth, P)
    end
    return nothing
end

create_path(str) = begin
    _path = split(str, '.')
    return Tuple([parse_path_elem(_str) for _str in _path])
end

parse_path_elem(str) = begin
    out = tryparse(Int, str)
    return isnothing(out) ? Symbol(str) : out+1
end

setpath!(nt::NamedTuple, path::Tuple, value) = begin
    return setpath!(getproperty(nt, path[1]), path[2:end], value)
end
setpath!(t::Tuple, path::Tuple, value) = begin
    !(path[1] isa Integer) && error("Expected first path entry to be integer! Got `path[1] = $(path[1])`") 
    return setpath!(t[path[1]], path[2:end], value)
end
setpath!(Y::AbstractArray, path::Tuple, value) = begin
    !isempty(path) && error("Last path not empty! Got `path = $(path)`")
    sz_Y, sz_v = size(Y), size(value) 
    prod(sz_Y) != prod(sz_v) && error("Size of field and value must be the same! Got `size(field) = $sz_Y` and `size(value) = $sz_v`")
    Y .= reshape(value, sz_Y)
    return nothing
end

function canonicalize(smiles; is_canonical=false)
    if !is_canonical
        @warn("""
        SMILES are not canonicalized by `MLPROP.jl`!
        Either ensure that the used SMILES are canonicalized or load `RDKitMinimalLib.jl` (not working on Windows).
        """)
    end
    return smiles
end