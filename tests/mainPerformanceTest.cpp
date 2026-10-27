#include "ut.hpp"
#include "../nnue/nnue.hpp"

void registerPerformanceEngineSuite();

int main(){
    if (!NNUE::activateEmbedded()) return 1;
    registerPerformanceEngineSuite();
}
