#include "driver.hpp"

#include "../ascii_utils.hpp"
#include "../engine/engine.hpp"
#include "../debug.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace driver {
namespace {

using chess::Board;
using engine::Engine;

// RENDERING

char pieceGlyph(uint8_t piece) noexcept {
    if (piece == Board::EMPTY) return '.';
    constexpr std::string_view glyphs = "PpNnBbRrQqKk";
    const int idx = ((piece & Board::MASK_PIECE_TYPE) - 1) * 2
                  + ((piece & Board::MASK_COLOR) == Board::BLACK);
    return idx < 12 ? glyphs[idx] : '?';
}

std::string boardToString(const Board& board) {
    static constexpr std::string_view files = "  a b c d e f g h\n";
    std::string s(files);
    for (int row = 7; row >= 0; --row) {
        const char r = static_cast<char>('1' + row);
        s += r; s += ' ';
        for (int col = 0; col < 8; ++col) { s += pieceGlyph(board.get(row, col)); s += ' '; }
        s += ' '; s += r; s += '\n';
    }
    s += files;
    return s;
}

// INPUT

[[noreturn]] void quit() {
    std::cout << "Thanks for playing!\n";
    std::exit(EXIT_SUCCESS);
}

// One whitespace-delimited token, or quit on EOF (closed pipe / Ctrl-D).
std::string readToken() {
    std::string tok;
    if (!(std::cin >> tok)) quit();
    return tok;
}

// Loop until a single char in [lo, hi] is entered.
char menuChoice(const char* prompt, char lo, char hi) {
    while (true) {
        std::cout << prompt << std::flush;
        const std::string in = readToken();
        if (in.size() == 1 && in[0] >= lo && in[0] <= hi) return in[0];
        std::cout << "Invalid choice.\n";
    }
}

// Prompt the human until they enter a legal move (or 'q' to quit).
void humanTurn(Engine& engine) {
    const Board& b = engine.board;
    std::cout << (b.getActiveColor() == Board::WHITE ? "\nWhite to move.\n" : "\nBlack to move.\n");
    while (true) {
        std::cout << "Move (e2e4 / e7e8q, 'q' to quit): " << std::flush;
        const std::string in = readToken();
        if (in == "q") quit();
        if (in.size() < 4 || in.size() > 5) { std::cout << "Bad format.\n"; continue; }

        const chess::Square from = chess::parseSquare(std::string_view(in).substr(0, 2));
        const chess::Square to   = chess::parseSquare(std::string_view(in).substr(2, 2));
        if (!chess::isValidSquare(from) || !chess::isValidSquare(to)) { std::cout << "Bad squares.\n"; continue; }

        char promo = (in.size() == 5) ? ascii::toLower(in[4]) : '\0';
        if (promo && !std::string_view("qrbn").contains(promo)) { std::cout << "Promote to q/r/b/n.\n"; continue; }
        // Auto-queen a pawn reaching the last rank when no piece was given.
        if (!promo) {
            const uint8_t p = b.get(from);
            const uint8_t lastRank = (p & Board::MASK_COLOR) == Board::WHITE ? 0 : 7;
            if ((p & Board::MASK_PIECE_TYPE) == Board::PAWN && chess::rank(to) == lastRank) promo = 'q';
        }

        if (engine.movePiece(from, to, promo)) return;
        std::cout << "Illegal move.\n";
    }
}

void engineTurn(Engine& engine) {
    std::cout << "\nEngine thinking...\n" << std::flush;
    DBG_TIMER_DECLARE(t); DBG_TIMER_START(t);
    engine.search(engine.DEFAULTDEPTH);
    DBG_TIMER_MS(t, "engine search");
    DBG_LOG_STREAM("[DEBUG] nodes: " << engine.searchRuntime.nodesSearched << '\n');
}

// GAME LOOP 

// whiteHuman / blackHuman select who drives each colour (White moves first).
void playGame(Engine& engine, bool whiteHuman, bool blackHuman) {
    while (!engine.isGameOver()) {
        std::cout << boardToString(engine.board);
        const bool human = (engine.board.getActiveColor() == Board::WHITE) ? whiteHuman : blackHuman;
        if (human) humanTurn(engine); else engineTurn(engine);
    }

    std::cout << boardToString(engine.board);
    if (engine.isMate())
        std::cout << "Checkmate - " << (engine.board.getActiveColor() == Board::WHITE ? "Black" : "White") << " wins.\n";
    else if (engine.isStalemate()) std::cout << "Stalemate - draw.\n";
    else                           std::cout << "Draw.\n";

    std::cout << "\nPress Enter to return to the menu..." << std::flush;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string line;
    std::getline(std::cin, line);
}

void clearScreen() {
#ifdef _WIN32
    [[maybe_unused]] const int r = std::system("cls");
#else
    [[maybe_unused]] const int r = std::system("clear");
#endif
}

[[noreturn]] void runMenu(Engine& engine, uci::UCI& uci) {
    while (true) {
        engine.reset();
        clearScreen();
        switch (menuChoice("\n=== HydraY ===\n"
                           "1) Play as White\n"
                           "2) Play as Black\n"
                           "3) Two players\n"
                           "4) Engine vs engine\n"
                           "5) UCI mode\n"
                           "6) Quit\n> ", '1', '6')) {
            case '1': playGame(engine, true,  false); break;
            case '2': playGame(engine, false, true);  break;
            case '3': playGame(engine, true,  true);  break;
            case '4': playGame(engine, false, false); break;
            case '5': uci.mainLoop(); // [[noreturn]]
            case '6': quit();
        }
    }
}

// CLI mode from argv[1]; returns to fall through to the interactive menu.
// `uci` and unknown args never return (mainLoop runs forever / usage exits).
void runCli(int argc, char* argv[], Engine& engine, uci::UCI& uci) {
    if (argc < 2) return;
    std::string mode = argv[1];
    for (char& c : mode) c = ascii::toLower(c);

    if (mode == "uci" || mode == "-uci" || mode == "--uci") uci.mainLoop();
    else if (mode == "-pvp") playGame(engine, true, true);
    else if (mode == "-bvb") playGame(engine, false, false);
    else if (mode == "-pvb") {
        const char side = (argc >= 3 && argv[2][0] && !argv[2][1]) ? ascii::toLower(argv[2][0]) : '\0';
        if (side != 'w' && side != 'b') {
            std::cout << "Specify a colour: -pvb w  or  -pvb b\n";
            std::exit(EXIT_FAILURE);
        }
        playGame(engine, side == 'w', side == 'b');
    } else {
        std::cout << "Usage: chess [-pvp | -bvb | -pvb w|b | uci]\n";
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

Driver::Driver(Engine& e) : engine_(e), uci_(e) {}

void Driver::startGame(int argc, char* argv[]) noexcept {
    runCli(argc, argv, engine_, uci_);
    runMenu(engine_, uci_);
}

} // namespace driver
