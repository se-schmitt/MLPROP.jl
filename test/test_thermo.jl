@testitem "Thermo calculations" begin
    using Clapeyron

    @testset "HANNA models" begin
        params_sat = (; A=[10.11564, 10.33675], B=[1687.537, 1648.22], C=[230.17, 230.918], 
            Tc=[647.13, 513.92], Pc=[2.19e7, 6.12e6], Tmin=[273.2, 276.5], Tmax=[473.2, 369.54])
        sat = AntoineEqSat(["water", "ethanol"]; userlocations=params_sat)
        model = HANNA(["water", "ethanol"]; puremodel=sat)
        model_og = ogHANNA(["water", "ethanol"]; puremodel=sat) 

        T = 350.
        p = 1e5

        @test first(dew_temperature(model, p, [1.,1.])) ≈ 357.3025375804405 rtol=1e-6
        @test first(dew_temperature(model_og, p, [1.,1.])) ≈ 357.1115867631476 rtol=1e-6
        @test first(bubble_pressure(model, T, [0.,1.])) ≈ 95797.11447554835 rtol=1e-6
        @test first(bubble_pressure(model_og, T, [0.,1.])) ≈ 95797.11447554835 rtol=1e-6
    end
end