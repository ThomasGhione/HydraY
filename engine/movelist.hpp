#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

#include "../board/board.hpp"

inline constexpr size_t MAX_MOVES = 218;

// Fixed-capacity move list over raw storage: a fresh list initialises no
// moves, and a copy moves only the live prefix.
struct MoveList {
    struct alignas(chess::Move) Storage { std::byte bytes[sizeof(chess::Move)]; };
    Storage data[MAX_MOVES];
    int size = 0;

    MoveList() noexcept = default;
    MoveList(const MoveList& other) noexcept { *this = other; }
    inline MoveList& operator=(const MoveList& other) noexcept {
        if (this != &other) {
            size = other.size;
            std::memcpy(data, other.data, static_cast<size_t>(size) * sizeof(chess::Move));
        }
        return *this;
    }

    template<typename... Args>
    inline void emplace_back(Args&&... args) noexcept {
        new (&data[size]) chess::Move(std::forward<Args>(args)...);
        ++size;
    }

    inline chess::Move& operator[](size_t i) noexcept { return begin()[i]; }
    inline const chess::Move& operator[](size_t i) const noexcept { return begin()[i]; }
    inline chess::Move* begin() noexcept { return reinterpret_cast<chess::Move*>(data); }
    inline chess::Move* end() noexcept { return begin() + size; }
    inline const chess::Move* begin() const noexcept { return reinterpret_cast<const chess::Move*>(data); }
    inline const chess::Move* end() const noexcept { return begin() + size; }
    [[nodiscard]] inline bool is_empty() const noexcept { return size == 0; }
};
