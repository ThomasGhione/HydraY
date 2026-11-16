#pragma once

namespace engine {

// `./chess bench [depth]`: the bench6 positions searched to a fixed depth
// (default 14) on one thread, each after a ucinewgame, so the node count is a
// deterministic fingerprint of the search. The last line is
// "<nodes> nodes <nps> nps", the format OpenBench reads. `bench 12` gives the
// same per-position counts as `script/engine_driver.sh bench6`.
int runBenchCommand(int argc, char* argv[]);

} // namespace engine
