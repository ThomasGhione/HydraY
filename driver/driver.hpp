#pragma once

#include "../uci/uci.hpp"

namespace engine { class Engine; }

namespace driver {

// Terminal front-end: CLI mode dispatch (-pvp/-bvb/-pvb/uci) and the
// interactive text menu. All game-loop logic lives file-local in driver.cpp;
// the class is just the owner of the engine reference and its UCI interface.
class Driver final {
public:
    explicit Driver(engine::Engine& engine);
    [[noreturn]] void startGame(int argc, char* argv[]) noexcept;

private:
    engine::Engine& engine_;
    uci::UCI uci_;
};

} // namespace driver
