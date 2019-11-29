"""
Author: Simon Lehnerer
"""

using Random


"""
Splits an array of indices into a training and test array

# Arguments
- `indices::AbstractVector`: the array of indices to split
- `train_size::Float64`: the relative size of the training set
- `shuffle::Bool=false`: if true, the array is shuffled before splitting

# Examples
```julia-repl
julia> train_test_split_indices(1:10, 0.8)
(1:8, 9:10)

julia> train_test_split_indices(1:10, 0.6, shuffle=true)
([1, 3, 4, 9, 6, 7], [8, 2, 10, 5])
```
"""
function train_test_split_indices(indices::AbstractVector, train_size::Float64; shuffle::Bool=false)
    
    if !(0 <= train_size <= 1)
        error("train_size must be in [0,1]")
    end
    
    n = length(indices)
    indices_copy = shuffle ? Random.shuffle(indices) : deepcopy(indices)
    split_index = round(Int, n * train_size)
    train_indices = split_index > 0 ? indices_copy[1:split_index] : []
    test_indices = split_index < n ? indices_copy[(split_index + 1):n] : []
    
    train_indices, test_indices
    
end


"""
Splits a range from 1 to n into a training and test array

# Arguments
- `n::Int64`: the end of the range
- `train_size::Float64`: the relative size of the training set
- `shuffle::Bool=false`: if true, the range is shuffled before splitting

# Examples
```julia-repl
julia> train_test_split_indices(10, 0.8)
(1:8, 9:10)

julia> train_test_split_indices(10, 0.6, shuffle=true)
([1, 3, 4, 9, 6, 7], [8, 2, 10, 5])
```
"""
function train_test_split_indices(n::Int64, train_size::Float64; shuffle::Bool=false)
    if n <= 0
        error("n must be positive")
    end
    train_test_split_indices(1:n, train_size, shuffle=shuffle)
end


"""
Splits a dataset (X,y) into training data and test data

# Arguments
- `X::AbstractArray`: stacked matrix of input variables
- `y::AbstractVector`: vector of output variables
- `train_size::Float64`: the relative size of the training set
- `shuffle::Bool=false`: if true, X and y are shuffled before splitting

# Examples
```julia-repl
julia> X = [1 2 3; 4 5 6; 7 8 9; 10 11 12]; y = [13, 14, 15, 16];
julia> train_test_split(X, y, 0.75)
([1 2 3; 4 5 6; 7 8 9], [10 11 12], [13, 14, 15], [16])
```
"""
function train_test_split(X::AbstractArray, y::AbstractVector, train_size::Float64; shuffle::Bool=false)
    
    if size(X)[1] != length(y)
        error("number of rows of X must be equal to the length of y")
    end
    if !(0 <= train_size <= 1)
        error("train_size must be in [0,1]")
    end
    
    train_indices, test_indices = train_test_split_indices(length(y), train_size, shuffle=shuffle)
    X_copy = deepcopy(X)
    y_copy = deepcopy(y)

    X_copy[train_indices, :], X_copy[test_indices, :], y_copy[train_indices], y_copy[test_indices]

end


"""
Returns a k-fold split of an array of indices

# Arguments
- `indices::AbstractVector`: the array to split
- `K::Int64`: the number of splits
- `shuffle::Bool=false`: if true, the array is shuffled before splitting

# Examples
```julia-repl
julia> indices = [7, 5, 4, 1, 2, 3];
julia> k_fold_indices(indices, 3)
([7, 5, 4, 1], [2, 3])
([7, 5, 2, 3], [4, 1])
([4, 1, 2, 3], [7, 5])
```
"""
function k_fold_indices(indices::AbstractVector, K::Int64; shuffle::Bool=false)

    if !(0 < K <= length(indices))
        error("K must be in [1, length(indices)]")
    end
     
    n = length(indices)
    indices_copy = shuffle ? Random.shuffle(indices) : deepcopy(indices)
    
    output = []
    
    for k in 1:K
        validation_indices = indices_copy[round(Int, (k-1)/K*n)+1:round(Int, k/K*n)]
        training_indices = setdiff(indices_copy, validation_indices)
        prepend!(output, [(training_indices, validation_indices)])
    end
    
    output
    
end


"""
Returns a k-fold split of a range from 1 to n

# Arguments
- `n::Int64`: the end of the range
- `K::Int64`: the number of splits
- `shuffle::Bool=false`: if true, the range is shuffled before splitting

# Examples
```julia-repl
julia> k_fold_indices(8, 4)
4-element Array{Any,1}:
 ([1, 2, 3, 4, 5, 6], 7:8)
 ([1, 2, 3, 4, 7, 8], 5:6)
 ([1, 2, 5, 6, 7, 8], 3:4)
 ([3, 4, 5, 6, 7, 8], 1:2)
```
"""
function k_fold_indices(n::Int64, K::Int64; shuffle::Bool=false)
    if n <= 0
        error("n must be positive")
    end
    k_fold_indices(1:n, K, shuffle=shuffle)
end
    

"""
Returns a k-fold split of the dataset (X,y)

# Arguments
- `X::AbstractArray`: stacked matrix of input variables
- `y::AbstractVector`: vector of output variables
- `K::Int64`: the number of splits
- `shuffle::Bool=false`: if true, X and y are shuffled before splitting

# Examples
```julia-repl
julia> X = [1 2 3; 4 5 6; 7 8 9; 10 11 12]; y = [13, 14, 15, 16];
julia> k_fold(X, y, 4)
([1 2 3; 4 5 6; 7 8 9], [10 11 12], [13, 14, 15], [16])
([1 2 3; 4 5 6; 10 11 12], [7 8 9], [13, 14, 16], [15])
([1 2 3; 7 8 9; 10 11 12], [4 5 6], [13, 15, 16], [14])
([4 5 6; 7 8 9; 10 11 12], [1 2 3], [14, 15, 16], [13])
```
"""
function k_fold(X::AbstractArray, y::AbstractVector, K::Int64; shuffle::Bool=false)
     
    if size(X)[1] != length(y)
        error("number of rows of X must be equal to the length of y")
    end
    if !(0 < K <= length(y))
        error("K must be in [1, length(y)]")
    end

    X_copy = deepcopy(X)
    y_copy = deepcopy(y)

    output = []
    
    for (training_indices, validation_indices) in k_fold_indices(length(y), K, shuffle=shuffle)
        X_train, X_validation = X_copy[training_indices, :], X_copy[validation_indices, :]
        y_train, y_validation = y_copy[training_indices], y_copy[validation_indices]
        append!(output, [(X_train, X_validation, y_train, y_validation)])
    end
    
    output
    
end


"""
Creates batches of an array of indices

# Arguments
- `indices::AbstractVector`: the array to batchify
- `K::Int64`: the length of one batch
- `shuffle::Bool=false`: if true, the array is shuffled before being batchified

# Examples
```julia-repl
julia> indices = 1:10;
julia> batchify_indices(indices, 3)
4-element Array{UnitRange{Int64},1}:
 1:3  
 4:6  
 7:9  
 10:10
```
"""
function batchify_indices(indices::AbstractVector, K::Int64; shuffle::Bool=false)

    n = length(indices)
    if n == 0
        error("indices must not be empty")
    end
    if !(1 <= K <= n)
        error("K must be in [1, length(indices)]")
    end

    indices_copy = shuffle ? Random.shuffle(indices) : deepcopy(indices)

    [indices_copy[i:min(i+K-1,n)] for i in 1:K:n]

end


"""
Creates batches of a range from 1 to n

# Arguments
- `n::Int64`: the end of the range
- `K::Int64`: the length of one batch
- `shuffle::Bool=false`: if true, the range is shuffled before being batchified

# Examples
```julia-repl
julia> batchify_indices(10, 3)
4-element Array{UnitRange{Int64},1}:
 1:3  
 4:6  
 7:9  
 10:10
```
"""
function batchify_indices(n::Int64, K::Int64; shuffle::Bool=false)
    if n < 1
        error("n must be positive")
    end
    batchify_indices(1:n, K; shuffle=shuffle)
end


"""
Creates batches for the dataset (X,y)

# Arguments
- `X::AbstractArray`: stacked matrix of input variables
- `y::AbstractVector`: vector of output variables
- `K::Int64`: the length of one batch
- `shuffle::Bool=false`: if true, the range is shuffled before being batchified

# Examples
```julia-repl
julia> X = [1 2 3; 4 5 6; 7 8 9; 10 11 12]; y = [13, 14, 15, 16];
julia> batchify(X, y, 2)
2-element Array{Array{Array{Int64,N} where N,1},1}:
 [[1 2 3; 4 5 6], [13, 14]]   
 [[7 8 9; 10 11 12], [15, 16]]
```
"""
function batchify(X::AbstractArray, y::AbstractVector, K::Int64; shuffle::Bool=false)

    if size(X)[1] != length(y)
        error("number of rows of X must be equal to the length of y")
    end
    if !(1 <= K <= length(y))
        error("K must be in [1, length(y)]")
    end

    X_copy = deepcopy(X)
    y_copy = deepcopy(y)

    [[X_copy[batch_indices, :], y_copy[batch_indices]] for batch_indices in batchify_indices(length(y), K, shuffle=shuffle)]

end
    

"""
Calculates the mean squared error (MSE)

# Arguments
- `y::AbstractVector`: vector of the correct target variables
- `f_y::AbstractVector`: vector of the predicted target variables

# Examples
```julia-repl
julia> y = [0, 1, 2]; f_y = [2, 0, 1];
julia> MSE(y, f_y)
2.0
```
"""
function MSE(y::AbstractVector, f_y::AbstractVector)
    if length(y) == 0 || length(f_y) == 0
        error("y and f_y must not be empty")
    end
    if length(y) != length(f_y)
        error("y and f_y must have the same length")
    end
    sum((y - f_y).^2) / length(y)
end


"""
Calculates the accuracy

# Arguments
- `y::AbstractVector`: vector of the correct target variables
- `f_y::AbstractVector`: vector of the predicted target variables

# Examples
```julia-repl
julia> y = [0, 1, 2, 3]; f_y = [0, 5, 2, 5];
julia> accuracy(y, f_y)
0.5
```
"""
function accuracy(y::AbstractVector, f_y::AbstractVector)
    if length(y) == 0 || length(f_y) == 0
        error("y and f_y must not be empty")
    end
    if length(y) != length(f_y)
        error("y and f_y must have the same length")
    end
    sum(y .== f_y) / length(y)
end


"""
Removes dimensions that only countain one entry from an array

# Arguments
- `A::AbstractArray`: the array

# Examples
```julia-repl
julia> A = [1 2 3;]
1×3 Array{Int64,2}:
 1  2  3

julia> squeeze(A)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    dropdims(A, dims=singleton_dims)
end


"""
Calculates the softmax of a vector x

# Arguments
- `x::AbstractVector`: the vector x

# Examples
```julia-repl
julia> x = [-2, 0, 2];
julia> softmax(x)
3-element Array{Float64,1}:
 0.015876239976466765
 0.11731042782619838 
 0.8668133321973349
julia> sum(softmax(x))
1.0
```
"""
function softmax(x::AbstractVector)
    if length(x) == 0
        error("x must not be empty")
    end
    e = exp.(x .- maximum(x))
    e / sum(e)
end


"""
Calculates the logistic function at x

# Arguments
- `x::Float64`: the value of x

# Examples
```julia-repl
julia> logistic(0.0)
0.5
```
"""
function logistic(x::Float64)
    1.0 / (1-0 + exp(-x))
end


"""
Calculates the derivative of the logistic function at x

# Arguments
- `x::Float64`: the value of x

# Examples
```julia-repl
julia> logistic_prime(0.0)
0.25
```
"""
function logistic_prime(x::Float64)
    logistic(x) * (1 - logistic(x))
end


"""
Calculates the outer product of two vectors

# Arguments
- `x::AbstractVector`: the first vector x
- `y::AbstractVector`: the second vector y

# Examples
```julia-repl
julia> x = [3, 2, 4]; y = [2, 4, 3];
julia> outer(x, y)
3×3 Array{Int64,2}:
 6  12   9
 4   8   6
 8  16  12
```
"""
function outer(x::AbstractVector, y::AbstractVector)
    x .* y'
end


"""
Defines a convergence criterion based on an array of previous accuracies. If there was no
improvement larger than a threshold in the last k rounds, return false

# Arguments
- `accuracies::AbstractVector`: the array of accuracies
- `number_iterations::UInt=10`: how many iterations should be taken into consideration
- `threshold::Float64=0.001`: minimal improvement that has to be achieved in the last k rounds

# Examples
```julia-repl
julia> accuracies = [0.90, 0.88, 0.93, 0.92, 0.92, 0.91, 0.92, 0.93];
julia> not_converged(accuracies, number_iterations=5, threshold=0.05)
false
julia> not_converged(accuracies, number_iterations=8, threshold=0.05)
true
julia> not_converged(accuracies, number_iterations=5, threshold=0.02)
true
```
"""
function not_converged(accuracies::AbstractVector; number_iterations::Int64=10, threshold::Float64=0.001)
    if number_iterations <= 0
        error("number_iterations must be positive")
    end
    if !(0 < threshold < 1)
        error("threshold must be in (0,1)")
    end
    if length(accuracies) >= number_iterations && (accuracies[end] - minimum(accuracies[end-(number_iterations-1):end-1]) < threshold)
        return false
    end
    true
end

nothing
