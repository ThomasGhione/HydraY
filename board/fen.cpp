#include "board.hpp"
#include "../ascii_utils.hpp"
#include <algorithm>
#include <charconv>
#include <sstream>

namespace chess {

namespace {
constexpr std::string_view PIECE_CHARS = "PNBRQKpnbrqk";
}

uint8_t Board::safeParseInt(const std::string& section, int min, int max, int defaultValue) {
    if (section.empty()) return defaultValue;
    int v = 0;
    auto res = std::from_chars(section.data(), section.data() + section.size(), v);
    return (res.ec == std::errc{} && res.ptr == section.data() + section.size()) ? std::clamp(v, min, max) : defaultValue;
}

void Board::fenToBoard(const std::string& fen) {
    std::istringstream fenStream(fen);
    std::string board, active, castling, ep, half, full;
    if (!(fenStream >> board >> active)) return;
    // Trailing fields are optional, as in Stockfish: an EPD line stops after the
    // en passant square, and missing fields read as "- - 0 1".
    fenStream >> castling >> ep >> half >> full;

    std::array<uint8_t, 64> parsedBoard{};
    {
        int rank = 7, file = 0;
        for (char c : board) {
            if (c == '/') { --rank; file = 0; }
            else if (ascii::isDigit(c)) file += c - '0';
            else {
                const size_t i = PIECE_CHARS.find(c);
                if (rank < 0 || file > 7 || i == std::string_view::npos) return;
                parsedBoard[(7 - rank) * 8 + file++] = static_cast<uint8_t>((i % 6 + 1) | (i < 6 ? WHITE : BLACK));
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

std::string placementFen(const std::array<char, 64>& squares) {
    std::string fen;
    for (int sq = 0; sq < 64; ++sq) {
        if (sq > 0 && sq % 8 == 0) fen += '/';
        if (squares[sq] != '.') fen += squares[sq];
        else if (!fen.empty() && ascii::isDigit(fen.back())) ++fen.back();
        else fen += '1';
    }
    return fen;
}

std::string Board::boardToFen() const {
    std::array<char, 64> squares;
    for (int sq = 0; sq < 64; ++sq) {
        const uint8_t p = chessboard[sq];
        squares[sq] = (p == EMPTY) ? '.' : PIECE_CHARS[(p & MASK_PIECE_TYPE) - 1 + ((p & MASK_COLOR) == BLACK ? 6 : 0)];
    }
    std::string fen = placementFen(squares);
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
