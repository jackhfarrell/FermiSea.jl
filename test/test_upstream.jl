@testset "upstream namespace" begin
    exported_names = Set(names(FermiSea))
    trixi_names = Set(names(Trixi))
    allowed = Set([:FermiSea])

    @test isempty(setdiff(intersect(exported_names, trixi_names), allowed))
    @test FermiSea.flux === Trixi.flux
    @test FermiSea.varnames === Trixi.varnames
    @test FermiSea.max_abs_speeds === Trixi.max_abs_speeds
end
