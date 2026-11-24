#pragma once

#include <array>
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

// FEN piece placement of 64 squares (0 = a8), each a piece letter or '.'.
inline std::string placementFen(const std::array<char, 64>& squares) {
    std::string fen;
    for (int sq = 0; sq < 64; ++sq) {
        if (sq > 0 && sq % 8 == 0) fen += '/';
        if (squares[sq] != '.') fen += squares[sq];
        else if (!fen.empty() && ascii::isDigit(fen.back())) ++fen.back();
        else fen += '1';
    }
    return fen;
}

} // namespace chess
