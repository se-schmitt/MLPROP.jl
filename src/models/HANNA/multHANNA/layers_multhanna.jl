#using Lux, ConcreteStructs, Random, LinearAlgebra

@concrete struct multHANNALux <: AbstractLuxContainerLayer{(:theta, :alpha, :phi)}
    theta
    alpha
    phi
end

@concrete struct LipschitzLinear <: AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    n_power_iterations::Int
end


function LipschitzLinear(in_dims::Int, out_dims::Int; n_power_iterations=2)
    return LipschitzLinear(in_dims, out_dims, n_power_iterations)
end

# initial
function Lux.initialparameters(rng::AbstractRNG, l::LipschitzLinear)
    return (
        # W_raw random initialization 
        weight = randn(rng, Float32, l.out_dims, l.in_dims) .* 0.1f0, 
        bias = zeros(Float32, l.out_dims),
        # ci starts at 4.0
        ci = [4.0f0] 
    )
end

function Lux.initialstates(rng::AbstractRNG, l::LipschitzLinear)
    u_init = randn(rng, Float32, l.out_dims)
    v_init = randn(rng, Float32, l.in_dims)
    
    return (
        u = u_init ./ norm(u_init),
        v = v_init ./ norm(v_init)
    )
end

function ((l::LipschitzLinear)(x, ps, st))
    W = ps.weight
    u = st.u
    v = st.v
    
    # # Power Iteration (only for training)
    # for _ in 1:l.n_power_iterations
    #     v_new = W' * u
    #     v = v_new ./ (norm(v_new) + 1e-12f0) 
        
    #     u_new = W * v
    #     u = u_new ./ (norm(u_new) + 1e-12f0)
    # end
    
    largest_sv = dot(u, W * v)
    
    W_normed = W ./ (largest_sv + 1e-12)
    
    softplus_ci = log(1.0f0 + exp(ps.ci[1]))
    
    W_scaled = W_normed .* softplus_ci
    
    y = W_scaled * x .+ ps.bias
    
    # state doesnt change
    #st_new = (u = u, v = v)
    
    return y, st
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