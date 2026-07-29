#pragma once

#include <string>
#include <string_view>
#include <cstdint>

#include "../ascii_utils.hpp"

namespace chess {

// Board convention: a8=0, b8=1, ..., h8=7, a7=8, ..., h1=63
using Square = uint8_t;

inline constexpr Square NO_SQUARE = 255;

inline constexpr uint8_t file(Square sq) noexcept { return sq & 7; }
inline constexpr uint8_t rank(Square sq) noexcept { return sq >> 3; }
inline constexpr bool isValidSquare(Square sq) noexcept { return sq < 64; }

inline constexpr Square squareFrom(uint8_t f, uint8_t r) noexcept {
    return (f < 8 && r < 8) ? static_cast<Square>(r * 8 + f) : NO_SQUARE;
}

inline Square parseSquare(std::string_view input) noexcept {
    if (input.length() != 2) [[unlikely]] return NO_SQUARE;

    const char fileChar = ascii::toLower(input[0]);
    const char rankChar = input[1];

    if (fileChar < 'a' || fileChar > 'h' || rankChar < '1' || rankChar > '8') [[unlikely]]
        return NO_SQUARE;

    return static_cast<Square>(('8' - rankChar) * 8 + (fileChar - 'a'));
}

inline std::string squareToString(Square sq) noexcept {
    if (!isValidSquare(sq)) [[unlikely]] return "??";
    return {static_cast<char>('a' + file(sq)), static_cast<char>('8' - rank(sq))};
}

} // namespace chess
