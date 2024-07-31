### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ bb3cab62-56f4-11ee-06e1-7b7ccfafe8a9
begin
	import Pkg
	Pkg.activate("..")
	
	using ReachabilityAnalysis
	using Plots
	using ControlSystemsBase
	using OffsetArrays
	using PlutoUI
	using LinearAlgebra
	using Polyhedra
	using Serialization
	using PlotlyJS
	# For heuristic 3
	using JuMP
	using HiGHS
	plotlyjs()
	TableOfContents()
end

# ╔═╡ 02e741a7-8610-4000-9296-ce60f773f17f
md"""
# Control with Measurement Uncertainty

This notebook studies how control is affected by measurement uncertainty caused by neural networks, and investigates the tradeoff between cost of running the network and reachable set size for the resulting closed-loop plant dynamics.
"""

# ╔═╡ d8481d61-e9e8-4066-a996-075eefdb4b9b
md"""
## Reachability for Noisy Discrete Linear Systems

In this section, we use types from the JuliaReach organization to develop a reachability algorithm for discrete linear systems with an additive noise term.  That is, the dynamics are of the form

``x[k+1] = A x[k] + w``

where ``w`` is an additive noise term bounded by a zonotope ``W``.
"""

# ╔═╡ 7138a285-0ba3-4fa8-8189-0f9ffdefdb79
"""
	reach(A, x0, W, H; max_order=Inf, reduced_order=2, remove_redundant=true)

Compute reachable sets for the dynamics ``x[k+1] = A x[k] + w``, where ``w`` is a noise term bounded by `W`.  The initial state is `x0`, and the time horizon is `H`.

If `max_order` is given, we reduce order of the reachable set to `reduced_order` when it exceeds this limit.  If `remove_redundant` is true, redundant generators are removed at each step.
"""
function reach(A::AbstractMatrix, x0::AbstractZonotope, W::AbstractZonotope, H::Integer; max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true)
	# Preallocate x vector
	x = OffsetArray(fill(x0, H+1), OffsetArrays.Origin(0))

	for k = 1:H
		x[k] = minkowski_sum(linear_map(A, x[k-1]), W)
		if remove_redundant
			x[k] = remove_redundant_generators(x[k])
		end
		if order(x[k]) > max_order
			x[k] = reduce_order(x[k], reduced_order)
		end
	end
	
	F = Flowpipe([ReachSet(x_k, k) for (k, x_k) in enumerate(x)])
end

# ╔═╡ 9d894075-9124-42b3-905e-38f0feb5a48f
md"""
To demonstrate this algorithm, we create the discrete-time dynamics of a five dimensional linear system.
"""

# ╔═╡ 55ddf867-1e6d-4031-b444-f78f6ab9c00f
A = let
D = [-1.0 -4.0 0.0 0.0 0.0;
     4.0 -1.0 0.0 0.0 0.0;
     0.0 0.0 -3.0 1.0 0.0;
     0.0 0.0 -1.0 -3.0 0.0;
     0.0 0.0 0.0 0.0 -2.0]
P = [0.6 -0.1 0.1 0.7 -0.2;
     -0.5 0.7 -0.1 -0.8 0.0;
     0.9 -0.5 0.3 -0.6 0.1;
     0.5 -0.7 0.5 0.6 0.3;
     0.8 0.7 0.6 -0.3 0.2]
P * D * inv(P)
end

# ╔═╡ e10a2d14-fec7-41d2-8ae5-acf442d7048a
ctrl_delay = 0.1

# ╔═╡ 0bd72276-5100-4c1d-843c-b6335734dc9f
Φ = ℯ^(A * ctrl_delay)

# ╔═╡ a39ef73d-093f-4cf5-b9e6-26f0b3352415
md"""
We then instantiate with an initial condition and a user-adjustable noise term.
"""

# ╔═╡ b688e2bc-9ffd-4c49-97d7-8c7632228d62
x0 = Zonotope([10., 10., 10., 10., 10.], collect(1. * I(5)))

# ╔═╡ 6685d33f-f5ba-46a0-b3bc-e481ccfa32e6
md"""Noise $(@bind noise Slider(0:0.01:0.1))"""

# ╔═╡ 5eec7abd-6c21-47e4-a5d6-af807fb751a7
W = Zonotope([0., 0., 0, 0, 0], noise * I(5))

# ╔═╡ 817af150-5516-4c49-9401-72d4c68ded68
md"""
Now we are ready to run the `reach` function, and plot its results.
"""

# ╔═╡ dceafb86-5583-47fc-af48-88edce5d57a4
r = reach(Φ, x0, W, 100)

# ╔═╡ 067936cf-b6f1-4392-9cd1-287338171c86
Plots.plot(r, vars=(4, 5))

# ╔═╡ d461425d-d88c-413b-b54d-5dc3e1172e65
md"""
Of course, we can analyze the solution as well, for instance by computing its diameter.
"""

# ╔═╡ 96f1cf81-d479-45e9-982b-cad1530e9048
[diameter(x.X) for x in r]

# ╔═╡ 1e8112d9-75b1-419d-872a-d80cba669f2a
maximum([diameter(x.X) for x in r])

# ╔═╡ dc34a7f4-2810-4a6f-8043-231e5741500d
md"""
## Exhaustive search over accuracy and cost

In this section, we examine the tradeoff between accuracy and cost of neural networks used for state sensing.  We begin by loading the accuracy and cost of running EfficientNet for image classification tasks, which we use as a model of accuracy vs. cost of state estimation.
"""

# ╔═╡ c2e067fc-05e4-4fdc-9655-dd4259b11c19
md"""
We map the accuracy in percent to a noise bound in the range ``[0, ∞)`` by taking ``\frac{100}{acc} - 1``.
"""

# ╔═╡ 6ab82ec9-c54a-4fcb-80b5-f0770ff4cff3
const dist_yolo_tradeoffs = (
    MobileNetv3_small = (err=47.66, gflop=9.869),
    # MobileNetv2 =       (err=44.31, gflop=43.786),
    ShuffleNetv2 =      (err=44.31, gflop=37.916),
    MobileNet =         (err=41.05, gflop=43.302),
    MobileNetv3_large = (err=32.90, gflop=43.731),
    Xception =          (err=26.91, gflop=104.002),
)

# ╔═╡ 97147c28-73ab-4ad9-b07c-8886085cdbf3
const tradeoff_map = vcat(([nn.err nn.gflop] for nn in dist_yolo_tradeoffs)...)

# ╔═╡ 12f96485-a85a-479d-aeb3-7a9a196b3e62
n = size(tradeoff_map, 1)

# ╔═╡ ffba1085-cce5-4b35-9271-97683e6da157
md"""
The next cell calculates an exhaustive search of all EfficientNet options for each state dimension of the plant.  For each assignment, we calculate the reachable set with EfficientNet's noise bounds (as computed above).

!!! warning
    Do not enable this cell unless you are prepared to run a ~30 minute calculation!
"""

# ╔═╡ c634038a-5ec2-4c6a-97af-d340617eef89
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
points = let
	points = [(0, 0, 0, 0, 0); Inf; Inf]
	for indices in Iterators.product([axes(efficient_net_map, 1) for _ in axes(Φ, 1)]...)
		W = Zonotope(zeros(axes(Φ, 1)), diagm(efficient_net_map[collect(indices), 2]))
		r = reach(Φ, x0, W, 100)
		md = maximum([diameter(x.X) for x in r])
		points = hcat(points, [indices; md; sum(efficient_net_map[collect(indices), 3])])
	end
	points
end
  ╠═╡ =#

# ╔═╡ 4af7631d-60c5-4739-8b42-da77fcff8701
md"""
We can then view the points as a table, or as a scatter plot.
"""

# ╔═╡ 9e143da7-bd8e-4123-8ac7-bcb4a93fe797
begin
	allpoints = deserialize("../data/dist_yolo_points.jls")
	allpoints = vcat(allpoints, [true fill(true, 1, size(allpoints, 2)-1)])
	points = allpoints[1:4, map(x -> reduce(&, x .<= n), allpoints[1,:])]
	# @info points
	# safepoints = points[1:3, BitVector(points[4,:])]
	# unsfpoints = points[1:3, BitVector(.!points[4,:])]
	# points = points[1:3, :]
end

# ╔═╡ 106b5ce1-053e-47c4-83f1-130274d727ed
begin
	nn_choices = 5
	ndims = 5
	diameters = zeros(fill(nn_choices, ndims)...)
	costs = zeros(fill(nn_choices, ndims)...)
	safetys = fill(false, fill(nn_choices, ndims)...)

	for point in eachcol(allpoints)
		coord = point[1]
		diameters[coord...] = point[2]
		costs[coord...] = point[3]
		safetys[coord...] = point[4]
	end

	get_safety(coord) = safetys[coord...]
	get_diameter(coord) = diameters[coord...]
	get_costs(coord) = costs[coord...]
	get_shape(coords, tshape, fshape) = ifelse.(get_safety.(coords), tshape, fshape)
	sep_points(points) = let
		safe = filter(point -> get_safety(point[1]), eachcol(points))
		unsf = filter(point -> !get_safety(point[1]), eachcol(points))
		return reduce(hcat, safe), reduce(hcat, unsf, init=fill(nothing, 4, 0))
	end
	safepoints, unsfpoints = sep_points(points)
end

# ╔═╡ 6a7be35e-ee7b-48a8-83a0-2e1c8163170a
begin
	Plots.scatter(unsfpoints[2,:], unsfpoints[3,:], label="Unsafe points", xlabel="Diameter", ylabel="Cost", hover=string.(unsfpoints[1,:]), markershape=:x, color=1)
	Plots.scatter!(safepoints[2,:], safepoints[3,:], label="Safe points", xlabel="Diameter", ylabel="Cost", hover=string.(safepoints[1,:]), color=1)
	
	# Plots.scatter(points[2,:], points[3,:], label="Exhaustive search", xlabel="Diameter", ylabel="Cost", hover=string.(points[1,:]), shape=get_shape(points[1,:], :o, :x), color=1)
end

# ╔═╡ 4f89e331-b945-4290-8d21-a2c108ee5d92
begin
	pts = filter(x -> x[1][1] == 3 && x[1][2] == 3 && x[1][3] == 3 && x[1][4] == 3, eachcol(points))
	Plots.scatter(points[2,:], points[3,:], label="Exhaustive search", hover=string.(points[1,:]))
	for p in pts
		Plots.scatter!([p[2]], [p[3]], label="", color=:red, hover=string(p[1]))
	end
	Plots.scatter!()
end

# ╔═╡ 82bf5d67-053a-4450-b7bc-dd7b02c86529
md"""
Furthermore, it is easy to extract the Pareto front from the data.
"""

# ╔═╡ ca409632-0ec3-4545-99f6-20d389a81f23
optimal = let
	optimal = [(0, 0, 0, 0, 0); Inf; Inf; false]
	for p in eachcol(points)
		add = true
		for q in eachcol(points)
			if p[2] > q[2] && p[3] >= q[3]
				add = false
			end
		end
		if add
			optimal = hcat(optimal, reshape(collect(p), 4, 1))
		end
	end
	optimal[:,2:end]
end

# ╔═╡ 3837f2d6-693f-46cb-8418-54d2a5f7e47e
Plots.scatter(optimal[2,:], optimal[3,:], label="Exhaustive search", xlabel="Diameter", ylabel="Cost", hover=string.(optimal[1,:]))

# ╔═╡ f70a5855-c799-43f6-ae1c-888c1ef2e85c
for o in eachcol(optimal)
	@info o
end

# ╔═╡ 44919be2-e2a6-485b-96ea-ac6c7eb708aa
dets = [abs(det(Φ[begin:end .!= i, begin:end .!= i])) for i in axes(Φ, 1)]

# ╔═╡ 06caa108-2156-40ce-bc32-97718c92f9e0
perm_mat = I(5)[sortperm(dets, by=i -> -i),:]

# ╔═╡ ee661a94-05a5-4023-b16d-4e78b90ec8a8
heur2 = let
	W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[[1, 1, 1, 1, 1], 1]))
	r = reach(Φ, x0, W, 100)
	reach_counter = 1
	md = maximum([diameter(x.X) for x in r])
	heur = [(1, 1, 1, 1, 1); md; sum(tradeoff_map[[1, 1, 1, 1, 1], 2])]
	for row in eachrow(repeat(perm_mat, size(tradeoff_map, 1) - 1, 1))
		# @info row heur[1, end]
		indices = row .+ collect(heur[1,end])
		# @info indices
		W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[indices, 1]))
		r = reach(Φ, x0, W, 100)
		reach_counter += 1
		md = maximum([diameter(x.X) for x in r])
		heur = hcat(heur, [tuple(indices...); md; sum(tradeoff_map[indices, 2])])
	end
	@info reach_counter
	heur
end

# ╔═╡ 8964836c-ca2d-4a1d-8565-6e542e20aa1b
md"""
## Heuristic search for accuracy/cost tradeoff

In this section, we develop a heuristic for extracting a near-optimial set of solutions faster than the exhaustive search.  This heuristic works by starting with the lowest cost option for all dimensions, then incrementing each dimension from the optimal solutions of the previous iteration in turn.  For example, after trying (1, 1) in the first round, we try (2, 1) and (1, 2) next.  If (2, 1) is found to be optimal, the next round tries (3, 1) and (2, 2), and so on.
"""

# ╔═╡ 27e7da0c-500a-432a-a2ca-b96a4da88436
heur, inte = let
	W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[[1, 1, 1, 1, 1], 1]))
	r = reach(Φ, x0, W, 100)
	md = maximum([diameter(x.X) for x in r])
	heur = [(1, 1, 1, 1, 1); md; sum(tradeoff_map[[1, 1, 1, 1, 1], 2])]
	inte = heur
	lastround = heur
	tried = Set([(1, 1, 1, 1, 1)])

	# Record # of calls to reachability function
	reach_counter = 1
	
	for round in length(heur[1,1])+1:(size(tradeoff_map, 1) * length(heur[1,1]))
		# Expand solution space
		thisround = [(0, 0, 0, 0, 0); Inf; Inf]
		# @info "Solutions from the last round" lastround
		# For each solution we found last round
		for h in eachcol(lastround)
			# Try incrementing the NN index for each NN
			for place in eachindex(h[1])
				inc = collect(h[1]) + [zeros(Int64, place - 1); 1; zeros(Int64, 	length(h[1]) - place)]
				for i in eachindex(inc)
					inc[i] = clamp(inc[i], axes(tradeoff_map, 1))
				end
				# @info "Incremented NN indices to" inc

				inct = tuple(inc...)
				if inct ∈ tried
					continue
				else
					union!(tried, [inct])
					# Don't allow too much difference in indices.  Speeds up the
					# heuristic by removing interior points, but also removes some
					# Pareto-optimal solutions.
					# if maximum(abs.(inc' .- inc)) > 1
					# 	continue
					# end
				end
				W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[inc, 1]))
				r = reach(Φ, x0, W, 100)
				reach_counter += 1
				md = maximum([diameter(x.X) for x in r])
				thisround = hcat(thisround, [inct; md; sum(tradeoff_map[inc, 2])])
			end
		end
		# Prune dominated solutions
		thisroundopt = [(0, 0, 0, 0, 0); Inf; Inf]
		thisroundint = [(0, 0, 0, 0, 0); Inf; Inf]
		for p in eachcol(thisround[:,begin+1:end])
			add = true
			for q in Iterators.flatten((eachcol(heur), eachcol(thisround[:,begin+1:end])))
				if p[2] > q[2] && p[3] >= q[3]
					add = false
				end
			end
			if add
				thisroundopt = hcat(thisroundopt, reshape(collect(p), 3, 1))
			else
				thisroundint = hcat(thisroundint, reshape(collect(p), 3, 1))
			end
		end
		heur = hcat(heur, thisroundopt[:,begin+1:end])
		inte = hcat(inte, thisroundint[:,begin+1:end])
		lastround = thisroundopt[:,begin+1:end]
	end
	@info reach_counter
	heur, inte
end

# ╔═╡ 8eb8896c-c54e-4cf6-b71d-945f21432b43
size(inte)

# ╔═╡ 271efb4d-b349-4531-b020-3a0c4106dae5
md"""
## Heuristic Using Sensitivity Analysis
"""

# ╔═╡ c82e1d58-759d-4208-8be0-bef20ee6abb4
heuristic3(sensitivity::Vector{<:Real}, budget::Real, tradeoff_map::Matrix{<:Real}) = let 
	cost(level) = tradeoff_map[level, 2]

	nlevels = size(tradeoff_map, 1)
	nstates = size(sensitivity, 1)

	model = Model(HiGHS.Optimizer)
	@variable(model, a[1:nstates,1:nlevels] >= 0, Int)
	@objective(model, Min, sum(a * tradeoff_map[:,1] .* sensitivity))
	@constraint(model, sum(a * tradeoff_map[:,2]) <= budget)
	for i in 1:nstates
		@constraint(model, sum(a[i,:]) == 1)
	end
	set_silent(model)
	optimize!(model)
	map(argmax, [value.(a)[i,:] for i in axes(value.(a),1)])
end

# ╔═╡ 05faadb9-28a7-48aa-b9b2-90d1010b6092
heur3 = let
	uniform_budgets = range(5*tradeoff_map[1,2], 5*tradeoff_map[n,2], 40)
	@info length(uniform_budgets)
	mul_sensitivity = [0.45, 0.29, 0.48, 0.68, 0.27]
	val = map(uniform_budgets) do budget
	    selection = heuristic3(mul_sensitivity, budget, tradeoff_map)
	end
	
	points = map(val) do selection
		W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[selection, 1]))
		r = reach(Φ, x0, W, 100)
		[Tuple(selection), maximum([diameter(x.X) for x in r]), sum(tradeoff_map[selection,2])]
	end
	hcat(points...)
end

# ╔═╡ c73b8dcf-73bd-43d2-9d5f-5ea680c1e169
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
all_budgets = let
	# Enumerate all possible selections
	all_selections = reshape(Iterators.product(fill(collect(1:n),5)...) |> collect, :)
	# Get cost (potential budget levels) for each selection
	all_budgets = map(x -> sum(efficient_net_map[collect(x),3]), all_selections)
	# Remove duplicate and sort
	Set(all_budgets) |> collect |> sort
end
  ╠═╡ =#

# ╔═╡ e95f6363-6d2f-4a88-aa30-139f7f594637
md"""
## Plots
"""

# ╔═╡ 0bfc7a46-88ee-4314-9f47-82db0449aba4
md"""Save plots? $(@bind toggle_saveplots CheckBox())"""

# ╔═╡ fa71efe9-ccfa-4075-85b5-c7e5297f99a0
size(optimal)

# ╔═╡ 843cb555-5ac2-4956-9245-65b6cdcdb847
begin
	# plt_es = Plots.scatter(
	# 	points[2,:], points[3,:], label="Exhaustive Search", 
	# 	xlabel="Diameter", ylabel="Cost",
	# 	xlabelfontsize=15,
	# 	ylabelfontsize=15,
	# 	xtickfontsize=12,
	# 	ytickfontsize=12,
	# 	legendfontsize=12,
	# 	# color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9)
	# )
	# scatter!(plt_es, optimal[2,:], optimal[3,:], 
	# 	label="Optimal solutions", 
	# )
	# Plots.savefig(plt_es, "../images/es.pdf")
	# plt_es

	
	plt_es = Plots.scatter(safepoints[2,:], safepoints[3,:], 
		label="Exhaustive Search", 
		xlabel="Maximum diameter of reachable sets", 
		ylabel="Cost (billion FLOPs)", 
		xlabelfontsize=15,
		ylabelfontsize=15,
		xtickfontsize=12,
		ytickfontsize=12,
		legendfontsize=12,
		# bottom_margin=0.1,
		# color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9)
	)
	Plots.scatter!(unsfpoints[2,:], unsfpoints[3,:], 
		label="Exhaustive Search (unsafe)",
		# color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9), 
		shape=:x,
		color=1
	)

	opsafe, opunsf = sep_points(optimal)
	Plots.scatter!(opsafe[2,:], opsafe[3,:], 
		label="Optimal solutions", 
		color=2
	)
	Plots.scatter!(opunsf[2,:], opunsf[3,:], 
		label="Optimal solutions (unsafe)", 
		shape=:x,
		color=2
	)
	if toggle_saveplots
		Plots.savefig(plt_es, "../images/es.pdf")
	end
	plt_es
end

# ╔═╡ a95d593c-3da8-4f09-9b99-a3d6020772dd
md"""
Plotting the optimal solutions, we can see that almost the entire Pareto front from above is found, along with a few interior points.  Removing these interior points may be one plan of attack for speeding up the heuristic further.
"""

# ╔═╡ 60e5b3a5-7aac-4eee-8366-a45b9bfb8210
begin
	# plt_dp = Plots.scatter(points[2,:], points[3,:], label="Exhaustive Search", xlabel="Diameter", ylabel="Cost", 
	# 	xlabelfontsize=15,
	# 	ylabelfontsize=15,
	# 	xtickfontsize=12,
	# 	ytickfontsize=12,
	# 	legendfontsize=12,
	# 	# shape=ifelse.(points[4,:], :o, :x),
	# 	color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9))
	# # plt = Plots.scatter(optimal[2,:], optimal[3,:], label="Optimal solutions", xlabel="Diameter", ylabel="Cost", hover=string.(optimal[1,:]))
	# Plots.scatter!(inte[2,:], inte[3,:], 
	# 	# shape=ifelse.(inte[4,:], :diamond, :x),
	# 	shape=:diamond,
	# 	label="Dynamic Programming (pruned)", 
	# 	color=3)
	# Plots.scatter!(heur[2,:], heur[3,:], 
	# 	# shape=ifelse.(heur[4,:], :diamond, :x),
	# 	shape=:diamond,
	# 	label="Dynamic Programming", 
	# 	color=2)
	# Plots.savefig(plt_dp, "../images/dp.pdf")
	# plt_dp

	plt_dp = Plots.scatter(safepoints[2,:], safepoints[3,:], 
		label="Exhaustive Search", 
		xlabel="Maximum diameter of reachable sets", 
		ylabel="Cost (billion FLOPs)",
		xlabelfontsize=15,
		ylabelfontsize=15,
		xtickfontsize=12,
		ytickfontsize=12,
		legendfontsize=12,
		color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9))
	# Plots.scatter!(unsfpoints[2,:], unsfpoints[3,:], label="Exhaustive Search (unsafe)",color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9), shape=:x)
	h1safe, h1unsf = sep_points(heur)
	i1safe, i1unsf = sep_points(inte)
	
	Plots.scatter!(i1safe[2,:], i1safe[3,:], 
		# shape=ifelse.(inte[4,:], :diamond, :x),
		shape=:diamond,
		label="Dynamic Programming (pruned)", 
		color=3)
	# Plots.scatter!(i1unsf[2,:], i1unsf[3,:], 
	# 	# shape=ifelse.(inte[4,:], :diamond, :x),
	# 	shape=:x,
	# 	label="Dynamic Programming (pruned, unsafe)", 
	# 	color=3)
	Plots.scatter!(h1safe[2,:], h1safe[3,:], 
		# shape=ifelse.(heur[4,:], :diamond, :x),
		shape=:diamond,
		label="Dynamic Programming", 
		color=2)
	# Plots.scatter!(h1unsf[2,:], h1unsf[3,:], 
	# 	# shape=ifelse.(heur[4,:], :diamond, :x),
	# 	shape=:x,
	# 	label="Dynamic Programming (unsafe)", 
	# 	color=2)
	if toggle_saveplots
		Plots.savefig(plt_dp, "../images/dp.pdf")
	end
	plt_dp
end

# ╔═╡ 53c3f6d9-4cfa-407a-8a17-50320eb97290
begin
	plt_fi = Plots.plot(
		label="Exhaustive Search", 
		xlabel="Maximum diameter of reachable sets", 
		ylabel="Cost (billion FLOPs)", 
		xlabelfontsize=15,
		ylabelfontsize=15,
		xtickfontsize=12,
		ytickfontsize=12,
		legendfontsize=12)
	Plots.scatter!(points[2,:], points[3,:], 
		label="Exhaustive Search",
		color=RGB(0.90,0.97,1), 
		markerstrokecolor=RGB(0.9,0.9,0.9), 
		shape=:o)
	# plt_fi = Plots.scatter(safepoints[2,:], safepoints[3,:], label="Exhaustive Search", xlabel="Maximum diameter of reachable sets", ylabel="Cost (billion FLOPs)", 
	# 	xlabelfontsize=15,
	# 	ylabelfontsize=15,
	# 	xtickfontsize=12,
	# 	ytickfontsize=12,
	# 	legendfontsize=12,
	# 	color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9))
	# Plots.scatter!(unsfpoints[2,:], unsfpoints[3,:], label="Exhaustive Search (unsafe)",color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9), shape=:x)
	
	# plt = Plots.scatter(optimal[2,:], optimal[3,:], label="Optimal solutions", xlabel="Diameter", ylabel="Cost", hover=string.(optimal[1,:]))
	
	# h2safe, h2unsf = sep_points(heur2)
	# Plots.scatter!(h2safe[2,:], h2safe[3,:], shape=:diamond, label="Fast Iterative", color=2)
	# Plots.scatter!(h2unsf[2,:], h2unsf[3,:], shape=:x, label="Fast Iterative (unsafe)", color=2)
	Plots.scatter!(heur2[2,:], heur2[3,:], shape=:diamond, label="Fast Iterative")
	if toggle_saveplots
		Plots.savefig(plt_fi, "../images/fi.pdf")
	end
	plt_fi
end

# ╔═╡ bdc43369-da5f-4950-b0df-a8c49814eaba
begin
	# plt_sv = Plots.scatter(points[2,:], points[3,:], label="Exhaustive Search", xlabel="Diameter", ylabel="Cost", 
	# 	xlabelfontsize=15,
	# 	ylabelfontsize=15,
	# 	xtickfontsize=12,
	# 	ytickfontsize=12,
	# 	legendfontsize=12,
	# 	color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9))
	# # plt = Plots.scatter(optimal[2,:], optimal[3,:], label="Optimal solutions", xlabel="Diameter", ylabel="Cost", hover=string.(optimal[1,:]))
	# Plots.scatter!(heur3[1,:], heur3[2,:], shape=:diamond, label="Sensitivity Analysis")
	# Plots.savefig(plt_sv, "../images/sv.pdf")
	# plt_sv

	plt_sv = Plots.scatter(safepoints[2,:], safepoints[3,:], 
		label="Exhaustive Search",
		xlabel="Maximum diameter of reachable sets", 
		ylabel="Cost (billion FLOPs)", 
		xlabelfontsize=15,
		ylabelfontsize=15,
		xtickfontsize=12,
		ytickfontsize=12,
		legendfontsize=12,
		color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9))
	Plots.scatter!(unsfpoints[2,:], unsfpoints[3,:], label="Exhaustive Search (unsafe)",color=RGB(0.90,0.97,1), markerstrokecolor=RGB(0.9,0.9,0.9), shape=:x)
	
	h3safe, h3unsf = sep_points(heur3)
	Plots.scatter!(h3safe[2,:], h3safe[3,:], shape=:diamond, label="Sensitivity Analysis", color=2)
	Plots.scatter!(h3unsf[2,:], h3unsf[3,:], shape=:x, label="Sensitivity Analysis (unsafe)", color=2)
	if toggle_saveplots
		Plots.savefig(plt_sv, "../images/sv.pdf")
	end
	plt_sv
end

# ╔═╡ 57f35365-759f-40a0-8a22-b8e9299c95db
md"""
## Illustration of diameter and cost over accuracy

In this section, we demonstrate the effects of accuracy of each dimension on diameter of the reachable set and cost of running the neural networks.  For this illustration, we use a two-dimensional version of the dynamics from the previous section.
"""

# ╔═╡ 2b394eb1-c251-4cbe-a56c-e9b49de34736
A_small = A[2:3,2:3]

# ╔═╡ 50538ed8-bc5c-4009-a797-dea6e93c57ce
Φ_small = ℯ^(A_small * ctrl_delay)

# ╔═╡ 2eeb108e-b69d-40f2-994b-ec4aab7159eb
x0_small = Zonotope([10., 10.], [1; 0;; 0; 1.])

# ╔═╡ aed6a9d8-6810-4b2c-a02e-68dbba357f89
begin
	diam_small = zeros(size(efficient_net_map_full, 1), size(efficient_net_map_full, 1))
	cost_small = zeros(size(efficient_net_map_full, 1), size(efficient_net_map_full, 1))
	for indices in Iterators.product([axes(efficient_net_map_full, 1) for _ in axes(Φ_small, 1)]...)
		W = Zonotope(zeros(axes(Φ_small, 1)), diagm(efficient_net_map_full[collect(indices), 2]))
		r = reach(Φ_small, x0_small, W, 100)
		diam_small[indices...] = maximum([diameter(x.X) for x in r])
		cost_small[indices...] = sum(efficient_net_map_full[collect(indices), 3])
	end
end

# ╔═╡ 2b89ba5c-6585-43d5-adf6-9f639768b031
Plots.heatmap(diam_small, xlabel="x_1 NN index", ylabel="x_2 NN index")

# ╔═╡ 208364c1-4c86-4f8b-9b94-7581ae41fa82
Plots.heatmap(cost_small, xlabel="x_1 NN index", ylabel="x_2 NN index")

# ╔═╡ 93efc2eb-54cc-46d0-80b6-52cf4104bc90
md"""
## Appendix

### Imports
"""

# ╔═╡ Cell order:
# ╟─02e741a7-8610-4000-9296-ce60f773f17f
# ╟─d8481d61-e9e8-4066-a996-075eefdb4b9b
# ╠═7138a285-0ba3-4fa8-8189-0f9ffdefdb79
# ╟─9d894075-9124-42b3-905e-38f0feb5a48f
# ╠═55ddf867-1e6d-4031-b444-f78f6ab9c00f
# ╠═e10a2d14-fec7-41d2-8ae5-acf442d7048a
# ╠═0bd72276-5100-4c1d-843c-b6335734dc9f
# ╟─a39ef73d-093f-4cf5-b9e6-26f0b3352415
# ╠═b688e2bc-9ffd-4c49-97d7-8c7632228d62
# ╠═5eec7abd-6c21-47e4-a5d6-af807fb751a7
# ╟─6685d33f-f5ba-46a0-b3bc-e481ccfa32e6
# ╟─817af150-5516-4c49-9401-72d4c68ded68
# ╠═dceafb86-5583-47fc-af48-88edce5d57a4
# ╠═067936cf-b6f1-4392-9cd1-287338171c86
# ╟─d461425d-d88c-413b-b54d-5dc3e1172e65
# ╠═96f1cf81-d479-45e9-982b-cad1530e9048
# ╠═1e8112d9-75b1-419d-872a-d80cba669f2a
# ╟─dc34a7f4-2810-4a6f-8043-231e5741500d
# ╟─c2e067fc-05e4-4fdc-9655-dd4259b11c19
# ╠═6ab82ec9-c54a-4fcb-80b5-f0770ff4cff3
# ╠═97147c28-73ab-4ad9-b07c-8886085cdbf3
# ╠═12f96485-a85a-479d-aeb3-7a9a196b3e62
# ╟─ffba1085-cce5-4b35-9271-97683e6da157
# ╠═c634038a-5ec2-4c6a-97af-d340617eef89
# ╟─4af7631d-60c5-4739-8b42-da77fcff8701
# ╠═9e143da7-bd8e-4123-8ac7-bcb4a93fe797
# ╠═106b5ce1-053e-47c4-83f1-130274d727ed
# ╠═6a7be35e-ee7b-48a8-83a0-2e1c8163170a
# ╠═4f89e331-b945-4290-8d21-a2c108ee5d92
# ╟─82bf5d67-053a-4450-b7bc-dd7b02c86529
# ╠═ca409632-0ec3-4545-99f6-20d389a81f23
# ╠═3837f2d6-693f-46cb-8418-54d2a5f7e47e
# ╠═f70a5855-c799-43f6-ae1c-888c1ef2e85c
# ╠═44919be2-e2a6-485b-96ea-ac6c7eb708aa
# ╠═06caa108-2156-40ce-bc32-97718c92f9e0
# ╠═ee661a94-05a5-4023-b16d-4e78b90ec8a8
# ╟─8964836c-ca2d-4a1d-8565-6e542e20aa1b
# ╠═27e7da0c-500a-432a-a2ca-b96a4da88436
# ╠═8eb8896c-c54e-4cf6-b71d-945f21432b43
# ╟─271efb4d-b349-4531-b020-3a0c4106dae5
# ╠═c82e1d58-759d-4208-8be0-bef20ee6abb4
# ╠═05faadb9-28a7-48aa-b9b2-90d1010b6092
# ╠═c73b8dcf-73bd-43d2-9d5f-5ea680c1e169
# ╟─e95f6363-6d2f-4a88-aa30-139f7f594637
# ╟─0bfc7a46-88ee-4314-9f47-82db0449aba4
# ╠═fa71efe9-ccfa-4075-85b5-c7e5297f99a0
# ╟─843cb555-5ac2-4956-9245-65b6cdcdb847
# ╟─a95d593c-3da8-4f09-9b99-a3d6020772dd
# ╟─60e5b3a5-7aac-4eee-8366-a45b9bfb8210
# ╟─53c3f6d9-4cfa-407a-8a17-50320eb97290
# ╟─bdc43369-da5f-4950-b0df-a8c49814eaba
# ╟─57f35365-759f-40a0-8a22-b8e9299c95db
# ╠═2b394eb1-c251-4cbe-a56c-e9b49de34736
# ╟─50538ed8-bc5c-4009-a797-dea6e93c57ce
# ╟─2eeb108e-b69d-40f2-994b-ec4aab7159eb
# ╠═aed6a9d8-6810-4b2c-a02e-68dbba357f89
# ╠═2b89ba5c-6585-43d5-adf6-9f639768b031
# ╠═208364c1-4c86-4f8b-9b94-7581ae41fa82
# ╟─93efc2eb-54cc-46d0-80b6-52cf4104bc90
# ╠═bb3cab62-56f4-11ee-06e1-7b7ccfafe8a9
