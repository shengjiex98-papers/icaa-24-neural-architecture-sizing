using ReachabilityAnalysis
using Plots
using ControlSystemsBase
using OffsetArrays
using LinearAlgebra
using Polyhedra
using Serialization

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
	
	Flowpipe([ReachSet(x_k, k) for (k, x_k) in enumerate(x)])
end

const A = let
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
const ctrl_delay = 0.1
const Φ = ℯ^(A * ctrl_delay)
const x0 = Zonotope(fill(10., 5), collect(1.0 * I(5)))

const dist_yolo_tradeoffs = (
    MobileNetv3_small = (err=47.66, gflop=9.869),
    MobileNetv3_large = (err=32.90, gflop=43.731),
    MobileNet =         (err=41.05, gflop=43.302),
    MobileNetv2 =       (err=44.31, gflop=43.786),
    ShuffleNetv2 =      (err=44.31, gflop=37.916),
    Xception =          (err=26.91, gflop=104.002),
)
const tradeoff_map = vcat(([nn.err nn.gflop] for nn in dist_yolo_tradeoffs)...)

function exhaustive_search(tradeoffmap, Φ, x0, all=true)
	upto = size(tradeoffmap, 1)
	points = [(0,0,0,0,0); Inf; Inf]
	for idx in Iterators.product(fill(axes(tradeoffmap, 1), size(Φ, 1))...)
		all || reduce(|, map(x -> x == upto, idx)) || continue
		W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoffmap[collect(idx), 1]))
		r = reach(Φ, x0, W, 100)
		md = maximum([diameter(x.X) for x in r])
		points = hcat(points, [idx; md; sum(tradeoffmap[collect(idx), 2])])
	end
	points[:,2:end]
end

@info "Start searching"
@time res = exhaustive_search(tradeoff_map, Φ, x0)
serialize("../data/exhaustive_points.jls", res)
