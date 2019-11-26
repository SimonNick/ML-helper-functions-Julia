using Random


"""
Splits a list of indices in a list of training indices and test indices

# Arguments
- `indices::AbstractVector`: the list of indices to split
- `train_size::Float64=0.80`: the relative size of the training set
- `shuffle::Bool=true`: whether to sample the indices randomly
"""
function train_test_split_indices(indices::AbstractVector; train_size::Float64=0.80, shuffle::Bool=true)
    
    @assert 0 <= train_size <= 1
    
    n = length(indices)
    indices_copy = shuffle ? Random.shuffle(indices) : deepcopy(indices)
    split_index = round(Int, n * train_size)
    train_indices = indices_copy[1:split_index]
    test_indices = indices_copy[(split_index + 1):n]
    
    return train_indices, test_indices
    
end


"""
Proxy for train_test_split_indices(1:n, train_size=train_size, shuffle=shuffle)
"""
function train_test_split_indices(n::Int64; train_size::Float64=0.80, shuffle::Bool=true)
    return train_test_split_indices(1:n, train_size=train_size, shuffle=shuffle)
end


"""
Splits a dataset (X,y) into training data and test data

# Arguments
- `X::AbstractArray`: stacked matrix of input variables
- `y::AbstractVector`: vector of target variables
- `train_size::Float64=0.80`: the relative size of the training set
- `shuffle::Bool=true`: whether to sample the dataset randomly
"""
function train_test_split(X::AbstractArray, y::AbstractVector; train_size::Float64=0.8, shuffle::Bool=true)
    
    @assert size(X)[1] == length(y)
    @assert 0 <= train_size <= 1
    
    train_indices, test_indices = train_test_split_indices(1:length(y), train_size, shuffle)
    X_copy = deepcopy(X)
    y_copy = deepcopy(y)
    X_copy[train_indices, :], X_copy[test_indices, :], y_copy[train_indices], y_copy[test_indices]

end


"""
Creates a set of k-fold training and validations sets of the indices

# Arguments
- `indices::AbstractVector`: the list of indices to split
- `K::Int64`: the number of splits
- `shuffle::Bool=true`: whether to sample the indices randomly
"""
function k_fold_indices(indices::AbstractVector, K::Int64; shuffle::Bool=false)

    @assert 0 < K <= length(indices)
     
    n = length(indices)
    indices_copy = shuffle ? Random.shuffle(indices) : deepcopy(indices)
    
    output = []
    
    for k in 1:K
        validation_indices = indices_copy[round(Int, (k-1)/K*n)+1:round(Int, k/K*n)]
        training_indices = indices_copy[setdiff(1:end, validation_indices)]
        append!(output, [(training_indices, validation_indices)])
    end
    
    output
    
end


"""
Proxy for k_fold_indices(1:n, K, shuffle=shuffle)
"""
function k_fold_indices(n::Int64, K::Int64; shuffle::Bool=false)
    k_fold_indices(1:n, K, shuffle=shuffle)
end
    

"""
Creates a set of k-fold training and validations sets of the dataset (X,y)

# Arguments
- `X::AbstractArray`: stacked matrix of input variables
- `y::AbstractVector`: vector of target variables
- `K::Int64`: the number of splits
- `shuffle=true::Bool`: whether to sample the dataset randomly
"""
function k_fold(X::AbstractArray, y::AbstractVector, K::Int64; shuffle::Bool=false)
     
    @assert size(X)[1] == length(y)
    @assert 0 < K <= length(y)

    X_copy = deepcopy(X)
    y_copy = deepcopy(y)
    
    for (training_indices, validation_indices) in 1:k_fold_indices(1:length(y), K, shuffle)
    
        X_train, X_validation = X_copy[setdiff(1:end, validation_indices), :], X_copy[validation_indices, :]
        y_train, y_validation = y_copy[setdiff(1:end, validation_indices)], y_copy[validation_indices]
        
        append!(output, [(X_train, X_validation, y_train, y_validation)])
    end
    
    output
    
end
    

"""
Calculates the mean squared error (MSE)

# Arguments
- `y::AbstractVector`: vector of the correct target variables
- `f_y::AbstractVector`: vector of the predicted target variables
"""
function MSE(y::AbstractVector, f_y::AbstractVector)
    sum((y - f_y).^2) / length(y)
end


"""
Calculates the accuracy

# Arguments
- `y::AbstractVector`: vector of the correct target variables
- `f_y::AbstractVector`: vector of the predicted target variables
"""
function accuracy(y::AbstractVector, f_y::AbstractVector)
    @assert length(y) == length(f_y)
    sum(y .== f_y) / length(y)
end


"""
Removes dimensions that only countain one entry from an array

# Arguments
- `A::AbstractArray`: the array
"""
function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end


"""
Calculates the softmax of a vector x

# Arguments
- `x::AbstractVector`: the vector x
"""
function softmax(x::AbstractVector)
    e = exp.(x .- maximum(x))
    e / sum(e)
end


"""
Defines a convergence criterion based on a list of previous accuracies. If there was no
improvement larger than a threshold in the last k rounds, return true

# Arguments
- `accuracies::AbstractVector`: the list of accuracies
- `number_iterations::UInt=10`: how many iterations should be taken into consideration
- `threshold::Float64=0.001`: minimal improvement that has to be achieved in the last k rounds
"""
function not_converged(accuracies::AbstractVector; number_iterations::Int64=10, threshold::Float64=0.001)
    @assert 0 < threshold < 1
    if length(accuracies) >= number_iterations && (accuracies[end] - minimum(accuracies[end-(number_iterations-1):end-1]) < threshold)
        return false
    end
    true
end