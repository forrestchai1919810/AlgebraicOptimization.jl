using LinearAlgebra, Random, Plots, Distributions

d = 2
N = 300
# x = rand(d, N)
M = 3
max_iterations = 1000
threshold = 0.0001

function makeData()
    groupOne = rand(MvNormal([10.0, 10.0], 10.0 * Matrix{Float64}(I, 2, 2)), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * Matrix{Float64}(I, 2, 2)), 100)
    groupThree = rand(MvNormal([15.0, 0.0], 10.0 * Matrix{Float64}(I, 2, 2)), 100)
    return hcat(groupOne, groupTwo, groupThree)'  # Transpose to get 300x2 matrix
end

x = makeData()'

μ = [rand(d) for _ in 1:M]
Σ = [Matrix{Float64}(I(d)) for _ in 1:M]
π = ones(M) / M
z = zeros(N, M)

function gaussian(x, μ, Σ)
    (2 * pi) ^ (-d / 2) * det(Σ) ^ (-1 / 2) * exp(-0.5 * (x - μ)' * inv(Σ) * (x - μ))
end

function e_step(x, μ, Σ, π, z)
    for i in 1:N
        for m in 1:M
            b = 0.0
            for j in 1:M
                b += gaussian(x[:, i], μ[j], Σ[j]) * π[j]
            end
            z[i, m] = gaussian(x[:, i], μ[m], Σ[m]) * π[m] / b
        end
    end
end

function m_step(x, μ, Σ, π, z)
    for m in 1:M
        numΣ = zeros(d, d)
        denΣ = 0.0
        numμ = zeros(d)

        for i in 1:N
            a = z[i, m] * (x[:, i] - μ[m]) * (x[:, i] - μ[m])'
            b = z[i, m]
            numΣ += a
            denΣ += b
            c = z[i, m] * x[:, i]
            numμ += c
        end

        Σ[m] = numΣ / denΣ
        μ[m] = numμ / denΣ
        π[m] = denΣ / N
    end
end

function diff(μ_old, μ_new)
    sum(norm(μ_old[i] - μ_new[i]) for i in 1:M)
end

function GMM(x,μ,Σ,π,z)
    ite = 0
    while true
        μ_old = deepcopy(μ)
        e_step(x, μ, Σ, π, z)
        m_step(x, μ, Σ, π, z)
        ite += 1
        if diff(μ_old, μ) < threshold || ite >= max_iterations
            break
        end
    end
end

GMM(x,μ,Σ,π,z)

function plot_results(x, μ, Σ, z)
    labels = [argmax(z[i, :]) for i in 1:N]
    scatter(x[1, :], x[2, :], group=labels, legend=false, title="MG Clustering Results")
    # scatter!(hcat(μ...), marker=:star, ms=10, label="Means")
end

plot_results(x, μ, Σ, z)
#savefig("MG_result.png")
