#include "board.hpp"
#include "../ascii_utils.hpp"
#include <algorithm>
#include <charconv>
#include <sstream>

namespace chess {

uint8_t Board::safeParseInt(const std::string& section, int min, int max, int defaultValue) {
    if (section.empty()) return defaultValue;
    int v = 0;
    auto res = std::from_chars(section.data(), section.data() + section.size(), v);
    return (res.ec == std::errc{} && res.ptr == section.data() + section.size()) ? std::clamp(v, min, max) : defaultValue;
}

void Board::fenToBoard(const std::string& fen) {
    std::istringstream fenStream(fen);
    std::string board, active, castling, ep, half, full;
    if (!(fenStream >> board >> active >> castling >> ep >> half >> full)) return;

    std::array<uint32_t, 8> parsedBoard{};
    {
        int rank = 7, file = 0;
        for (char c : board) {
            if (c == '/') { --rank; file = 0; }
            else if (ascii::isDigit(c)) file += c - '0';
            else {
                if (rank < 0 || file > 7) return;
                const uint8_t p = CHAR_TO_PIECE_TYPE[static_cast<uint8_t>(c)];
                if (p == EMPTY) return;
                parsedBoard[rank] |= static_cast<uint32_t>(p) << (file++ * 4);
            }
        }
        if (rank != 0 || file != 8) return;
    }

    chessboard = parsedBoard;
    activeColor = (!active.empty() && (active[0] == 'b' || active[0] == 'B')) ? BLACK : WHITE;

    castle = 0;
    for (char c : castling)
        if (const auto i = std::string_view("KQkq").find(c); i != std::string_view::npos)
            castle |= uint8_t(1 << i);

    enPassant = parseSquare(ep);
    halfMoveClock = safeParseInt(half, 0, 255, 0);
    fullMoveClock = safeParseInt(full, 1, 255, 1);

    rebuildBitboardsFromSquares();
    rebuildRepetitionHistory();
}

std::string Board::boardToFen() const {
    constexpr std::array<char, 8> pieceChar = {'.', 'P', 'N', 'B', 'R', 'Q', 'K', '?'};
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int emptySq = 0;
        for (int file = 0; file < 8; ++file) {
            const uint8_t p = (chessboard[rank] >> (file * 4)) & MASK_PIECE;
            if (p == EMPTY) { ++emptySq; continue; }
            if (emptySq) { fen += char('0' + emptySq); emptySq = 0; }
            const char sym = pieceChar[p & MASK_PIECE_TYPE];
            fen += (p & MASK_COLOR) == BLACK ? char(sym | 0x20) : sym;
        }
        if (emptySq) fen += char('0' + emptySq);
        if (rank > 0) fen += '/';
    }
    fen += activeColor == WHITE ? " w " : " b ";
    const size_t castleStart = fen.size();
    for (int i = 0; i < 4; ++i)
        if (castle & (1 << i)) fen += "KQkq"[i];
    if (fen.size() == castleStart) fen += '-';
    if (!isValidSquare(enPassant)) fen += " -";
    else { fen += ' '; fen += squareToString(enPassant); }
    fen += ' ';
    fen += std::to_string(halfMoveClock);
    fen += ' ';
    fen += std::to_string(fullMoveClock);
    return fen;
}

}
