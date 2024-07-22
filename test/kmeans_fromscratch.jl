using LinearAlgebra, Random, Plots, Distributions

d = 2
N = 300
M = 3
max_iterations = 100
threshold = 0.0001

function makeData()
    groupOne = rand(MvNormal([10.0, 10.0], 10.0 * Matrix{Float64}(I, 2, 2)), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * Matrix{Float64}(I, 2, 2)), 100)
    groupThree = rand(MvNormal([15.0, 0.0], 10.0 * Matrix{Float64}(I, 2, 2)), 100)
    return hcat(groupOne, groupTwo, groupThree)'  # Transpose to get 300x2 matrix
end

x = makeData()'

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

function plot_results_kmeans(x, μ, z)
    labels = [argmax(z[i, :]) for i in 1:N]
    scatter(x[1, :], x[2, :], group=labels, legend=false, title="K-means Clustering Results")
    # scatter!(hcat(μ...), marker=:star, ms=10, label="Means")
end

plot_results_kmeans(x, μ, z)
savefig("kmeans_result.png")
