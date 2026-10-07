function small_unstructured_mesh(; nx = 2, ny = 2)
    path = tempname() * ".mesh"
    node(i, j) = j * (nx + 1) + i + 1
    element(i, j) = j * nx + i + 1
    points = [(2 * i / nx - 1, 2 * j / ny - 1) for j in 0:ny for i in 0:nx]
    surfaces = NTuple{6,Int}[]
    for j in 0:ny, i in 0:(nx - 1)
        lower = j == 0 ? element(i, 0) : element(i, j - 1)
        upper = j == 0 || j == ny ? 0 : element(i, j)
        push!(surfaces, (node(i, j), node(i + 1, j), lower, upper,
                         j == 0 ? 1 : 3, j == ny ? 0 : 1))
    end
    for i in 0:nx, j in 0:(ny - 1)
        left = i == 0 ? element(0, j) : element(i - 1, j)
        right = i == 0 || i == nx ? 0 : element(i, j)
        push!(surfaces, (node(i, j), node(i, j + 1), left, right,
                         i == 0 ? 4 : 2, i == nx ? 0 : 4))
    end
    open(path, "w") do io
        println(io, "ISM-V2")
        println(io, length(points), " ", length(surfaces), " ", nx * ny, " 1")
        foreach(point -> println(io, point[1], " ", point[2], " 0.0"), points)
        foreach(surface -> println(io, join(surface, " ")), surfaces)
        for j in 0:(ny - 1), i in 0:(nx - 1)
            println(io, join((node(i, j), node(i + 1, j),
                              node(i + 1, j + 1), node(i, j + 1)), " "))
            println(io, "0 0 0 0")
            println(io, j == 0 ? "wall_bottom" : "---", " ",
                    i == nx - 1 ? "right" : "---", " ",
                    j == ny - 1 ? "wall_top" : "---", " ",
                    i == 0 ? "left" : "---")
        end
    end
    return path
end
