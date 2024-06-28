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
    MNv3s = (err=47.66, gflop=9.869),
    SNV2 =  (err=40.19, gflop=37.916),
    MNv3l = (err=32.91, gflop=43.731),
    # MN =    (err=41.05, gflop=43.302),
    # MNv2 =  (err=44.31, gflop=43.786),
    # B0 =    (err=35.31, gflop=54.051),
    # B1 =    (err=38.90, gflop=56.830),
    B2 =    (err=30.61, gflop=69.371),
    B3 =    (err=27.23, gflop=84.574),
    X =     (err=26.92, gflop=104.00),
    # B4 =    (err=28.13, gflop=118.589),
    # B5 =    (err=36.25, gflop=156.735),
    B6 =    (err=21.08, gflop=205.171),
    # B7 =    (err=21.11, gflop=269.646),
)
const tradeoff_map = vcat(([nn.err nn.gflop] for nn in dist_yolo_tradeoffs)...)

function exhaustive_search(tradeoff_map, Φ, x0, all=true)
    upto = size(tradeoff_map, 1)
    points = [(0,0,0,0,0); Inf; Inf]
    for idx in Iterators.product(fill(axes(tradeoff_map, 1), size(Φ, 1))...)
        all || reduce(|, map(x -> x == upto, idx)) || continue
        W = Zonotope(zeros(axes(Φ, 1)), diagm(tradeoff_map[collect(idx), 1]))
        r = reach(Φ, x0, W, 100)
        md = maximum([diameter(x.X) for x in r])
        points = hcat(points, [idx; md; sum(tradeoff_map[collect(idx), 2])])
    end
    points[:,2:end]
end

@info "Start searching"
@time res = exhaustive_search(tradeoff_map, Φ, x0)
serialize("../data/dist_yolo_points_7nn.jls", res)
