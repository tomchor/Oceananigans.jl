struct Foo
    bar
    baz
end

struct OrderedPair
    x::Real
    y::Real
    OrderedPair(x,y) = x > y ? error("out of order") : new(x,y)
end

function test(a::FT, b::FT)::Float64 where FT
    println(a, b, FT)
    return 4
end
