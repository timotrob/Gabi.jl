type Knn
  k::Int
  x
  x_norm
  y
  Knn() = new(3)
end


function normalize(input_df)
    n_cols = size(input_df)[2]
    n_rows = size(input_df)[1]
    norm_df = deepcopy(input_df)
    for i in 1:n_cols
        delta = maximum(input_df[:,i]) - minimum(input_df[:,i])
        minCol =  minimum(input_df[:,i])
        for j in 1:n_rows
            norm_df[j,i] = (norm_df[j,i] - minCol)/delta
        end
    end

    norm_df
end


function train!{U<:Real,T}(model::Knn, x::Matrix{U}, y::Vector{T})
  model.x = x
  model.x_norm = normalize(x)
  model.y = y
end

function predict{U<:Real}(model::Knn,x::Matrix{U})
    x_norm = normalize(x)
    println("x_norm",x_norm)
    if (eltype(model.y)<:Real)
       #Regression
        resp = Float64[size(x,1)]
        for i in 1:size(x_norm,1)
          println("predict instance",i)
          resp[i]= assign_value(model.x_norm, model.y, model.k, x_norm[i,:])
        end
        return resp
    else
      error("Classification is not implemented yet")
    end
end



function euclidean_distance(a, b)
  println("a:",a)
  println("b:",b)
 distance = 0.0
 for index in 1:size(a, 1)
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

function get_k_nearest_neighbors(xTrain, imageI, k)
  println("get_k_nearest_neighbors")
   nRows, nCols = size(xTrain)
   imageJ = Array(Float32, nRows)
   distances = Array(Float32, nCols)
   for j in 1:nCols
    for index in 1:nRows
     imageJ[index] = xTrain[index, j]
    end
    distances[j] = euclidean_distance(imageI, imageJ)
   end
   sortedNeighbors = sortperm(distances)
   kNearestNeighbors = sortedNeighbors[1:k]
   return kNearestNeighbors
end


function assign_value(xTrain, yTrain, k, imageI)
  println("assign_value")
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
