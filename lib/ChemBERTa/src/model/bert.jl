# Config
@kwdef struct BERTConfig
    vocab_size::Int
    emb_dim::Int
    max_pos_size::Int = 512
    typ_voc_size::Int = 2
    n_heads::Int
    n_layers::Int
    hidden_dim::Int
    dropout_rate::Float32 = 0f0
    act::Function = NNlib.gelu_erf
    dtype::Type = Float32
    position_offset::Int = 0            #TODO better padding
end

# Embedding
@concrete struct BERTEmbedding <: AbstractLuxContainerLayer{(:word_embed, :pos_embed, :type_embed, :norm, :dropout)}
    word_embed
    pos_embed
    type_embed
    norm
    dropout
    cfg::BERTConfig
end

function BERTEmbedding(cfg::BERTConfig)
    return BERTEmbedding(
        Embedding(cfg.vocab_size => cfg.emb_dim),
        Embedding(cfg.max_pos_size => cfg.emb_dim),
        Embedding(cfg.typ_voc_size => cfg.emb_dim),
        LayerNorm((cfg.emb_dim,); dims=nothing, epsilon=1f-12),
        Dropout(cfg.dropout_rate),
        cfg,
    )
end

function (emb::BERTEmbedding)(in::AbstractArray, ps, st::NamedTuple)
    seq_len = size(in, 1)
    B = ndims(in) > 1 ? size(in, 2) : 1
    in_type = ones(Int, seq_len, B)
    return emb((in, in_type), ps, st)
end

function (emb::BERTEmbedding)((in, in_type)::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple)
    seq_len = size(in, 1)
    B = ndims(in) > 1 ? size(in, 2) : 1

    word_emb, st_word = emb.word_embed(in, ps.word_embed, st.word_embed)

    in_pos = repeat((1:seq_len) .+ emb.cfg.position_offset, 1, B)
    pos_emb, st_pos = emb.pos_embed(in_pos, ps.pos_embed, st.pos_embed)

    type_emb, st_type = emb.type_embed(in_type, ps.type_embed, st.type_embed)

    combined = word_emb .+ pos_emb .+ type_emb
    normed, st_norm = emb.norm(combined, ps.norm, st.norm)
    out, st_drop = emb.dropout(normed, ps.dropout, st.dropout)

    st_out = (; word_embed=st_word, pos_embed=st_pos, type_embed=st_type, norm=st_norm, dropout=st_drop)
    return out, st_out
end

# Encoder
function BERTEncoderBlock(cfg::BERTConfig)
    return TransformerEncoderBlock(; 
        in_dim=cfg.emb_dim, 
        hidden_dims=(cfg.hidden_dim,), 
        nheads=cfg.n_heads, 
        act=cfg.act, 
        dropout_rate=cfg.dropout_rate,
        dense_kwargs=(; use_bias=true),
    )
end

# Main BERT Model
@concrete struct BERT <: AbstractLuxContainerLayer{(:embedding, :blocks)}
    embedding
    blocks
    cfg::BERTConfig
end

function BERT(cfg::BERTConfig)
    return BERT(
        BERTEmbedding(cfg),
        Tuple([BERTEncoderBlock(cfg) for _ in 1:(cfg.n_layers)]),
        cfg,
    )
end

function (bert::BERT)(in_idx::AbstractArray, ps, st::NamedTuple; mask=nothing)
    x, st_embedding = bert.embedding(in_idx, ps.embedding, st.embedding)

    st_blocks = ()
    for (i, block) in enumerate(bert.blocks)
        x, st_block_new = block((x, mask), ps.blocks[i], st.blocks[i])
        st_blocks = (st_blocks..., st_block_new)
    end

    return (
        x,
        (;
            embedding=st_embedding,
            blocks=st_blocks,
        ),
    )
end

function (bert::BERT)((in_idx, in_type)::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple; mask=nothing)
    x, st_embedding = bert.embedding((in_idx, in_type), ps.embedding, st.embedding)

    st_blocks = ()
    for (i, block) in enumerate(bert.blocks)
        x, st_block_new = block((x, mask), ps.blocks[i], st.blocks[i])
        st_blocks = (st_blocks..., st_block_new)
    end

    return (
        x,
        (;
            embedding=st_embedding,
            blocks=st_blocks,
        ),
    )
end

struct ChemBERTaModel
    smodel::StatefulLuxLayer
    tokenizer::ChemBERTaTokenizer
end
function (model::ChemBERTaModel)(smiles::AbstractString)
    # Canonicalize smiles
    _smiles = canonicalize(smiles)

    # tokenizer
    enc = encode(model.tokenizer.encoder, _smiles)
    input_ids = getindex.(findall(enc.token), 1)
    
    # ChemBERTa
    output = model.smodel(input_ids)
    
    return output[:,1,1]
end