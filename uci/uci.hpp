#pragma once

#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

namespace engine {
    class Engine;
}

namespace uci {

class UCI {
public:
    explicit UCI(engine::Engine& engine);
    ~UCI() noexcept;

    [[noreturn]] void mainLoop() noexcept;
    void parseCommand(const std::string& command) noexcept;

private:
    engine::Engine& engine;
    std::thread searchThread;
    std::mutex searchMutex;
    std::string searchBestMove = "0000";
    bool searchPonder = false;
    bool searchDone = false;
    bool searchPrinted = true;

    void uci() noexcept;
    void setOption(std::istringstream& is) noexcept;
    void position(std::istringstream& is) noexcept;
    void go(std::istringstream& is) noexcept;
    void ponderhit() noexcept;
    void stopSearch(bool printBestMove) noexcept;
    void emitBestMove(std::string_view move) noexcept; // caller must hold searchMutex
};

}
