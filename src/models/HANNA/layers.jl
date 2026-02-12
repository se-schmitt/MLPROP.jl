@concrete struct LuxHANNA <: AbstractLuxContainerLayer{(:theta,:alpha,:phi)}
    theta
    alpha
    phi
end

function (model::LuxHANNA)((T,x,embs), ps, st)
    θs = first(model.theta(embs, ps.theta, st.theta))
    
    # Calculate cosine similarity and distance between the two components
    θ1 = @view θs[:,1]
    θ2 = @view θs[:,2]
    
    cosine_sim_ij = cosine_similarity(θ1,θ2)
    cosine_dist_ij = 1.0 - cosine_sim_ij

    # Concatenate embeddings with T and x
    α1_in = vcat(T, x[1], θ1) 
    α2_in = vcat(T, x[2], θ2)

    α1_out = first(model(α1_in, ps.alpha, st.alpha))
    α2_out = first(model(α2_in, ps.alpha, st.alpha))
    
    c_mix = α1_out .+ α2_out
    
    gE_NN = first(model.phi(c_mix, ps.phi, st.phi))[1]

    return x[1]*x[2]*gE_NN*cosine_dist_ij
end