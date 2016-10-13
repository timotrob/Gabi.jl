type Knn
  k::Int
  x
  y
  means_train
  std_train
  norm_fx
  Knn() = new(3)
end

function train!{U<:Real,T}(model::Knn, x::Matrix{U}, y::Vector{T})
  model.x = x
  model.y = y
  nRows, nCols = size(x)
  model.means_train = Vector{Float64}(nCols)
  model.std_train = Vector{Float64}(nCols)
  for j in 1:nCols
      model.means_train[j] = mean(x[:,j])
      model.std_train[j] = std(x[:,j])
  end
end

function predict{U<:Real}(model::Knn,x::Matrix{U})
    nRows, nCols = size(x)
    x_norm = deepcopy(x)
    for j in 1:nCols
      for i in  1:nRows
         x_norm[i,j] = (x[i,j] - model.means_train[j])/(model.std_train[j])
      end
    end
    if (eltype(model.y)<:Real)
       #Regression
        resp = Vector{Float64}(nRows)
        for i in 1:size(x_norm,1)
          resp[i]= assign_value(x_norm, model.y, model.k, x_norm[i,:])
        end
        return resp
    else
      error("Classification is not implemented yet")
    end
end

function euclidean_distance(a, b)
 distance = 0.0
 for index in 1:size(a, 1)
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end


function get_k_nearest_neighbors(xTrain, rowTestItem, k)
   nRows, nCols = size(xTrain)
   distances = Array(Float32,nRows)
    for index in 1:nRows
     distances[index] = euclidean_distance(xTrain[index,:], rowTestItem)
    end
   sortedNeighbors = sortperm(distances)
   kNearestNeighbors = sortedNeighbors[1:k]
   return kNearestNeighbors
end

# assign_value(model.x_norm, model.y, model.k, x_norm[i,:])
function assign_value(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k)

 soma::Float64 = 0
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  soma = soma + yTrain[n]
 end
 return soma / k
end

function assign_label(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k)
 counts = Dict{Int, Int}()
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = yTrain[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 #add one to the count
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end
 end
 return mostPopularLabel
end
