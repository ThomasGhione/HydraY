// omp.h for clang builds without libomp: HydraY only calls this function,
// which libgomp provides; `#pragma omp simd` needs no runtime (-fopenmp-simd).
#pragma once

extern "C" int omp_get_max_threads(void) noexcept;
