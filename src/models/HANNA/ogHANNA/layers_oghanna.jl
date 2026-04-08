@concrete struct ogHANNALux <: AbstractLuxContainerLayer{(:theta,:alpha,:phi)}
    theta
    alpha
    phi
end

Clapeyron.is_splittable(::ogHANNALux) = false

function (model::ogHANNALux)((T,x,embs), ps, st)

    θs = [first(model.theta(_emb, ps.theta, st.theta)) for _emb in embs]
    
    # Calculate cosine similarity and distance between the two components
    cosine_sim_ij = cosine_similarity(θs[1],θs[2])
    cosine_dist_ij = 1.0 - cosine_sim_ij

    # Concatenate embeddings with T and x
    α1_in = vcat(T, x[1], θs[1]) 
    α2_in = vcat(T, x[2], θs[2])

    α1_out = first(model.alpha(α1_in, ps.alpha, st.alpha))
    α2_out = first(model.alpha(α2_in, ps.alpha, st.alpha))
    
    c_mix = α1_out .+ α2_out
    
    gE_NN = first(model.phi(c_mix, ps.phi, st.phi))[1]

    return x[1]*x[2]*gE_NN*cosine_dist_ij
end