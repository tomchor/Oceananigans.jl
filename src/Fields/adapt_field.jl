using Adapt

Adapt.adapt_structure(to, field::Field{X, Y, Z}) where {X, Y, Z} =
    Field{X, Y, Z}(Adapt.adapt(to, field.data),
                   Adapt.adapt(to, field.grid),
                   Adapt.adapt(to, field.boundary_conditions))