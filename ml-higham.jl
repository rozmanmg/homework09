"""
    sigmoid(x) = 1 / (1 + exp(-x))

Classic sigmoid activation function.
"""
@inline function sigmoid(x)
    t = exp(-abs(x))
    ifelse(x >= 0, inv(1 + t), t / (1 + t))
end

"""
    activate!(a, x, W, b)

Evaluate sigmoid function,

    a .= sigmoid.(W*x .+ b).

Here x is the input vector,  a is the output vector,
W it the matrix of the weights, b is the vector of biases. 
"""
@inline function activate!(a, x, W, b)
    a .= W * x .+ b
    # a .= muladd(W, x, b)
    a .= sigmoid.(a)
    return nothing
end

"""
    forward!(p, xp)

Forward pass through neural network
"""
function forward!(p, xp)
    W, B, wa = p
    l = length(wa) 

    wa[1] .= xp 
    for j = 2:l
        activate!(wa[j], wa[j-1], W[j-1], B[j-1])
    end
    return nothing
end

"""
    backward!(delta, p, yy)

Backward pass through neural network
"""
function backward!(delta, p, yy)
    W, _, wa = p
    l = length(wa) 

    delta[l] .= wa[l] .* (1.0 .- wa[l]) .* (wa[l] .- yy)
    for j = (l-1):-1:2
        delta[j] .= wa[j] .* (1.0 .- wa[j]) .* (W[j]' * delta[j+1])
    end
    return nothing
end

"""
    cost2(x, y, p)

Calculate the cost function  
"""
@views function cost2(x, y, p)

    s = 0.0
    # xp = zeros(size(x,1))
    
    for i = 1:size(x,2)
        forward!(p, x[:, i])
        s += (norm(y[:, i] .- p.wa[end], 2))^2
    end

    return s
end

"""
    predict2(xy, p)

"""
function predict2(xy, p)
    forward!(p, xy)
    return p.wa[end] ./ sum(p.wa[end]) # probability
end

"""
    fbpropagate2!(savecosts, p, delta, eta, niters, x, y)

Minimize cost function using forward and back propagate
"""
@views function fbpropagate2!(savecosts, p, delta, eta, niters, x, y)

    np = size(x, 2)
    
    W, B, wa = p
    l = length(wa) 

    @showprogress for i = 1:niters
        k = rand(1:np)
        
        # Forward pass
        forward!(p, x[:, k])

        # Backward pass
        backward!(delta, p, y[:, k])

        # Gradient step

        wa[1] .= x[:, k]
        for j = 2:l
            W[j - 1] .-= eta * delta[j] * wa[j-1]'
            B[j - 1] .-= eta * delta[j]
        end
        
        savecosts[i] = cost2(x, y, p)    
    end

    return nothing
end

function model_init(Ns)
    # Weights
    W = [0.5 * randn(Ns[i], Ns[i-1]) for i = 2:length(Ns)]
    
    # Biases
    B = [0.5 * randn(Ns[i]) for i = 2:length(Ns)]
    
    # Work arrays
    wa = [Vector{Float64}(undef, Ns[i]) for i = 1:length(Ns)]
    
    p = (; W, B, wa)
    
    # Deltas
    delta = [Vector{Float64}(undef, Ns[i]) for i = 1:length(Ns)]

    return p, delta
end

