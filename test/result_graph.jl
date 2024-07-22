using Plots

confusion_matrix_kmeans = [2685 1026; 990 6474]
confusion_matrix_gmm = [3082 818; 593 6682]

metrics = ["Mutual Information Score", "Mean Squared Error"]
values_kmeans = [7.6091475790069625, 0.9347543351977371]
values_gmm = [7.756825868493754, 1.3077388369647562]

bar(
    ["K-means TP", "K-means FP", "K-means FN", "K-means TN", "GMM TP", "GMM FP", "GMM FN", "GMM TN"],
    vcat(vec(confusion_matrix_kmeans), vec(confusion_matrix_gmm)),
    title="Confusion Matrix Comparison",
    legend=false,
    xlabel="Confusion Matrix Elements",
    ylabel="Count",
    xticks=(1:8, ["K-means TP", "K-means FP", "K-means FN", "K-means TN", "GMM TP", "GMM FP", "GMM FN", "GMM TN"]),
    bar_width=0.7
)
savefig("confusion_matrix_comparison.png")

bar(
    metrics,
    [values_kmeans values_gmm],
    title="Mutual Information Score and Mean Squared Error Comparison",
    label=["K-means" "GMM"],
    xlabel="Metrics",
    ylabel="Values",
    bar_width=0.7,
    legend=:topright
)
savefig("metrics_comparison.png")
