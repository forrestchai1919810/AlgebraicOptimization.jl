using Random
using Statistics
using Plots

function test_sort_times()
    sizes = [10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8]
    runtimes = Float64[]

    for size in sizes
        s = rand(size)
        push!(runtimes, @elapsed sort(s))
    end

    return sizes, runtimes
end

sizes, runtimes = test_sort_times()

plot(sizes, runtimes, 
    xlabel = "List Size", 
    ylabel = "Runtime (seconds)", 
    title = "Sort Runtime vs List Size",
    xscale = :log10, 
    yscale = :log10, 
    marker = :o, 
    legend = false)

println("Sizes: ", sizes)
println("Runtimes: ", runtimes)

savefig("sort_time.png")

display(plot)