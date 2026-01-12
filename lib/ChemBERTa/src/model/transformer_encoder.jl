@concrete struct TransformerEncoderBlock <: AbstractLuxContainerLayer{(:mha,:mlp,:mha_norm,:mlp_norm,:dropout)}
    mha
    mlp
    mha_norm
    mlp_norm
    dropout
end

function TransformerEncoderBlock(; 
    in_dim, 
    out_dim=in_dim, 
    hidden_dims=(), 
    nheads, 
    act=relu, 
    dropout_rate=0f0,
    dense_kwargs=(;),
)
    _dims1 = (in_dim, hidden_dims...)
    _dims2 = (hidden_dims..., out_dim)
    return TransformerEncoderBlock(
        MultiHeadAttention(
            in_dim; 
            nheads, 
            attention_dropout_probability=dropout_rate,
            dense_kwargs=dense_kwargs,
        ),
        Chain([
            Dense(dim1, dim2, i == length(_dims1) ? identity : act; dense_kwargs...) 
            for (i, (dim1,dim2)) in enumerate(zip(_dims1,_dims2))
        ]...),
        LayerNorm((in_dim,); dims=nothing, epsilon=1f-12),
        LayerNorm((out_dim,); dims=nothing, epsilon=1f-12),
        Dropout(dropout_rate),
    )
end

function (model::TransformerEncoderBlock)((x, mask), ps, st::NamedTuple)
    # attention block (+ shortcut connection)
    shortcut = x
    (attn_out, α), st_mha = model.mha((x, x, x, mask), ps.mha, st.mha)
    x = attn_out .+ shortcut
    x, st_mha_norm = model.mha_norm(x, ps.mha_norm, st.mha_norm)

    # feed-forward block (+ shortcut connection)
    shortcut = x
    x, st_mlp = model.mlp(x, ps.mlp, st.mlp)
    x = x .+ shortcut
    x, st_mlp_norm = model.mlp_norm(x, ps.mlp_norm, st.mlp_norm)
    
    # dropout 
    x, st_dropout = model.dropout(x, ps.dropout, st.dropout)

    return x, (;
        mha=st_mha,
        mlp=st_mlp,
        mha_norm=st_mha_norm,
        mlp_norm=st_mlp_norm,
        dropout=st_dropout,
    )
end