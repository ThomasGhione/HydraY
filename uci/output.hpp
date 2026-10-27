#pragma once

#include <iostream>
#include <mutex>
#include <string_view>

namespace uci {

// stdout has two writers in UCI mode: the command reader (readyok, option
// replies, bestmove on stop) and the search thread (info, bestmove). main()
// turns stdio sync off, so std::cout has no lock of its own: a `readyok`
// written while the search thread is still inside its `bestmove` flush gets
// the old buffer re-sent ahead of it, or is dropped when that flush resets
// the buffer. Every write goes through here as one complete, flushed block.
inline void emit(std::string_view text) noexcept {
    static std::mutex outputMutex;
    const std::lock_guard<std::mutex> lock(outputMutex);
    std::cout << text;
    std::cout.flush();
}

} // namespace uci
