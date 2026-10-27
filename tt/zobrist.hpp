#pragma once

#include <array>
#include <cstdint>

#include "../board/board.hpp"

namespace zobrist {

struct Tables {
    // By the board's piece nibble (type | 0x8 for white) and square.
    std::array<std::array<uint64_t, 64>, 16> pieces;
    uint64_t sideToMove;
    std::array<uint64_t, 16> castling;   // by the 4-bit KQkq rights mask
    std::array<uint64_t, 8> enPassant;   // by file
};

// Fixed-seed xorshift64*, run at compile time.
constexpr Tables makeTables() noexcept {
    Tables t{};
    uint64_t state = 0x123456789ABCDEF0ULL;
    const auto next = [&state] {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    };
    for (auto& squares : t.pieces)
        for (uint64_t& key : squares) key = next();
    t.sideToMove = next();
    for (uint64_t& key : t.castling) key = next();
    for (uint64_t& key : t.enPassant) key = next();
    return t;
}

inline constexpr Tables TABLES = makeTables();

// The en-passant file is hashed only when a pawn of the side to move can
// pseudo-legally capture there.
inline bool hasPseudoLegalEnPassantCapture(const chess::Board& board, chess::Square epSquare) noexcept {
    if (!chess::isValidSquare(epSquare)) return false;
    const int side = chess::Board::colorToIndex(board.getActiveColor());
    return (pieces::PAWN_ATTACKS[side ^ 1][epSquare] & board.bb[chess::Board::PAWN][side]) != 0ULL;
}

inline uint64_t computeHashKey(const chess::Board& board) noexcept {
    uint64_t key = TABLES.castling[board.castlingRights()];
    for (uint8_t sq = 0; sq < 64; ++sq)
        if (const uint8_t piece = board.get(sq)) key ^= TABLES.pieces[piece][sq];
    if (board.getActiveColor() == chess::Board::BLACK) key ^= TABLES.sideToMove;
    if (hasPseudoLegalEnPassantCapture(board, board.getEnPassant()))
        key ^= TABLES.enPassant[chess::file(board.getEnPassant())];
    return key;
}

} // namespace zobrist
