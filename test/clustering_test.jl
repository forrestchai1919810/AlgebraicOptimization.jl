using Clustering

# make a random dataset with 1000 random 5-dimensional points
X = rand(5, 1000)

# cluster X into 20 clusters using K-means
R = kmeans(X, 20; maxiter=200, display=:iter)

@assert nclusters(R) == 20 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers

using RDatasets, Clustering, Plots

iris = dataset("datasets", "iris")

features = collect(Matrix(iris[:, 1:4])')

result = kmeans(features, 3)

scatter(iris.PetalLength, iris.PetalWidth, 
        marker_z=result.assignments, 
        color=:lightrainbow, 
        legend=false,
        xlabel="Petal Length",
        ylabel="Petal Width",
        title="K-means Clustering of Iris Data")

savefig("kmeans_iris.png")

display(plot)

