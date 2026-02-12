include("layers.jl")
include("HANNA_legacy/hanna_legacy.jl")

# Utils 
silu(x) = @. x/(1+exp(-x))

function cosine_similarity(x1,x2;eps=1e-8)
    ∑x1 = sqrt(dot(x1,x1))
    ∑x2 = sqrt(dot(x2,x2))
    return dot(x1,x2)/(max(∑x1,eps*one(∑x1))*max(∑x2,eps*one(∑x2)))
end