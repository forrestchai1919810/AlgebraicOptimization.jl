using LinearAlgebra, Random, Plots, Distributions, CSV, DataFrames, Clustering, StatsBase

file_path = "/Users/ForrestChai/Desktop/julia_projects/AlgebraicOptimization.jl/iris.data"
data = CSV.read(file_path, DataFrame, header=false, normalizenames=true)

function convert_to_float(col)
    parse.(Float64, replace.(string.(col), "?" => "NaN"))
end

label_map = Dict("Iris-setosa" => 1, "Iris-versicolor" => 2, "Iris-virginica" => 3)
ground_truth = map(label -> label_map[label], data[!, 5])

for col in names(data)[1:4]
    data[!, col] = convert_to_float(data[!, col])
end

data_matrix = hcat([Vector{Float64}(data[!, col]) for col in names(data)[1:4]]...)
function normalize_data(data_matrix)
    mu = mean(data_matrix, dims=1)
    sigma = std(data_matrix, dims=1)
    normalized_data = (data_matrix .- mu) ./ sigma
    return normalized_data
end
x = normalize_data(data_matrix)'

d, N = size(x)
M = 3
max_iterations = 1000000
threshold = 0.000001

μ = [rand(d) for _ in 1:M]
z = zeros(N, M)

function e_step_kmeans(x, μ, z)
    for i in 1:N
        distances = [norm(x[:, i] - μ[m]) for m in 1:M]
        mindis = argmin(distances)
        z[i, :] .= 0
        z[i, mindis] = 1
    end
end

function m_step_kmeans(x, μ, z)
    for m in 1:M
        one_indices = findall(z[:, m] .== 1)
        if length(one_indices) > 0
            μ[m] = vec(sum(x[:, one_indices], dims=2) / length(one_indices))
        end
    end
end

function diff_kmeans(μ_old, μ_new)
    sum(norm(μ_old[i] - μ_new[i]) for i in 1:M)
end

function kmeans(x, μ, z)
    ite = 0
    while true
        μ_old = deepcopy(μ)
        e_step_kmeans(x, μ, z)
        m_step_kmeans(x, μ, z)
        ite += 1
        if diff_kmeans(μ_old, μ) < threshold || ite >= max_iterations
            break
        end
    end
end

kmeans(x, μ, z)

function plot_results_kmeans(x, μ, z, var1::Int, var2::Int)
    labels = [argmax(z[i, :]) for i in 1:N]
    scatter(x[var1, :], x[var2, :], group=labels, legend=false, title="K-means Clustering Results")
    # scatter!(hcat(μ...)', marker=:star, ms=10, label="Means")
end

plot_results_kmeans(x, μ, z, 3, 4)
savefig("kmeans_result.png")

assign_vector = [argmax(z[i, :]) for i in 1:N]
confusion_metric = confusion(assign_vector, ground_truth)
println("Confusion Matrix:")
println(confusion_metric)


labels = [argmax(z[i, :]) for i in 1:N]

function mutual_info_score(assign_vector, ground_truth)
    contingency = StatsBase.countmap(zip(assign_vector, ground_truth))
    N = sum(values(contingency))
    mutual_info = 0.0
    
    for ((i, j), nij) in contingency
        pi = sum(values(StatsBase.countmap(filter(x -> x[1] == i, keys(contingency)))))
        pj = sum(values(StatsBase.countmap(filter(x -> x[2] == j, keys(contingency)))))
        mutual_info += (nij / N) * log((N * nij) / (pi * pj))
    end
    
    return mutual_info
end

mutual_info = mutual_info_score(assign_vector, ground_truth)
println("Mutual Information Score: ", mutual_info)

function mean_squared_error(x, labels, centroids)
    mse = 0.0
    for i in 1:N
        mse += norm(x[:, i] - centroids[labels[i]])^2
    end
    return mse / N
end

mse = mean_squared_error(x, labels, μ)
println("Mean Squared Error: ", mse)
