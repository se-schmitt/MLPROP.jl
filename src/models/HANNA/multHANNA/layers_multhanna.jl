#using Lux, ConcreteStructs, Random, LinearAlgebra

@concrete struct multHANNALux <: AbstractLuxContainerLayer{(:theta, :alpha, :phi)}
    theta
    alpha
    phi
end

function (model::multHANNALux)((T, x, embs), gamma, ps, st)
    N = length(x)
    # theta input
    θs = first(model.theta(embs, ps.theta, st.theta)) # Output: (96, N)
    
    rbf_sim = [exp(-gamma * sum(abs2, θs[:, i] .- θs[:, j])) for i in 1:N, j in 1:N]
    
    x_adj = [sum(x[j] * rbf_sim[i, j] for j in 1:N) for i in 1:N]
    
    gE_total = zero(eltype(x)) 
    
    for i in 1:N
        for j in (i+1):N
            # Muggianu
            X_i_ij = (1.0 + x_adj[i] - x_adj[j]) / 2.0
            X_j_ij = (1.0 + x_adj[j] - x_adj[i]) / 2.0
            
            # Alpha input, adding pair interaction and temperature
            c_i = vcat(θs[:, i], X_i_ij, T)
            c_j = vcat(θs[:, j], X_j_ij, T)
            
            α_i = first(model.alpha(c_i, ps.alpha, st.alpha))
            α_j = first(model.alpha(c_j, ps.alpha, st.alpha))
            α_ij = α_i .+ α_j 
            
            # phi
            gE_NN_ij = first(model.phi(α_ij, ps.phi, st.phi))[1]
            
            # check simalarity
            correction = x[i] * x[j] * (1.0 - rbf_sim[i, j])
            
            # adding correction 
            gE_total += gE_NN_ij * correction
        end
    end
    
    return gE_total
end