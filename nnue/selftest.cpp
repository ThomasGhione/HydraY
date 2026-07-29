#include "selftest.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include "../board/board.hpp"
#include "../board/piece.hpp"
#include "../engine/sort/move_generator.hpp"
#include "accumulator.hpp"
#include "nnue.hpp"

namespace NNUE {

namespace {

bool accumulatorMatchesScratch(chess::Board& b, const char* context, int game, int ply) {
    // HalfKA laziness: a perspective whose king crossed bucket/flip is dirty
    // until the next evaluate(); settle it before comparing (this is exactly
    // the state evaluate() would see).
    b.ensureNnueAccumulatorClean();
    const Accumulator incremental = b.nnueAccumulator;
    b.refreshNnueAccumulator(); // overwrites with the from-scratch values
    if (std::memcmp(&incremental, &b.nnueAccumulator, sizeof(Accumulator)) != 0) {
        std::cout << "MISMATCH (" << context << ") game " << game << " ply " << ply
                  << "\n  fen: " << b.boardToFen() << "\n";
        return false;
    }
    return true;
}

} // namespace

int runSelfTest(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: ./chess nnue-selftest <net.bin> [games]\n";
        return 1;
    }
    pieces::initMagicBitboards();
    if (!loadNetwork(argv[2])) return 1;
    const int games = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 200;

    // Reference evals: must match nnue/trainer/src/bin/sanity.rs on the same
    // net. The mirrored pair must be exactly equal (perspective encoding).
    struct Ref { const char* label; const char* fen; };
    const Ref refs[] = {
        {"startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
        {"startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"},
        {"wN e4 extra (w)", "rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
        {"bN e5 extra (b)", "rnbqkbnr/pppppppp/8/4n3/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"},
        {"stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
        {"stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"},
        // Reduced-material refs: exercise different OUTPUT buckets (the six
        // above all sit in the top material bucket).
        {"middlegame ~24 pieces (w)", "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
        {"KRPvKR (w)", "8/8/4k3/8/2r5/2K5/4P3/3R4 w - - 0 1"},
        {"KQvK (w)", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"},
        // King-placement refs: exercise INPUT buckets and the mirror fold.
        // Keep in sync with trainer/src/bin/sanity.rs.
        {"both castled short (w)", "r4rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 1"},
        {"white long castle vs e8 king (w)", "r3kb1r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPPBQPPP/2KR3R w kq - 0 1"},
        {"kings on 2nd rank (w)", "8/1k3ppp/1p6/p1p5/P1P5/1P4P1/1K3P1P/8 w - - 0 1"},
        {"active kings endgame (w)", "8/8/3k4/8/2p5/2P1K3/8/8 w - - 0 1"},
    };
    int32_t mirrorPair[2] = {0, 0};
    for (size_t i = 0; i < std::size(refs); ++i) {
        chess::Board b(refs[i].fen);
        const int32_t cp = evaluate(b);
        if (i == 2 || i == 3) mirrorPair[i - 2] = cp;
        std::cout << refs[i].label << ": " << cp << " cp (stm)\n";
    }
    if (mirrorPair[0] != mirrorPair[1]) {
        std::cout << "FAIL: mirrored pair differs (" << mirrorPair[0] << " vs "
                  << mirrorPair[1] << ")\n";
        return 1;
    }

    // Random walks: after every doMove (and a doMove+undoMove probe of a
    // different random move) the incremental accumulator must be byte-equal
    // to a from-scratch rebuild. Covers captures/EP/castling/promotions.
    std::mt19937_64 rng(0xC0FFEE);
    uint64_t positions = 0;
    for (int game = 0; game < games; ++game) {
        chess::Board b{};
        for (int ply = 0; ply < 220; ++ply) {
            const MoveList legal = engine::MoveGenerator::generateLegalMoves(b);
            if (legal.is_empty() || b.isFiftyMoveRule() || b.hasInsufficientMaterialDraw()) break;

            chess::Board::MoveState st{};
            const chess::Move& probe = legal[rng() % static_cast<uint64_t>(legal.size)];
            b.doMove(probe, st);
            b.undoMove(probe, st);
            if (!accumulatorMatchesScratch(b, "do+undo", game, ply)) return 1;

            const chess::Move& m = legal[rng() % static_cast<uint64_t>(legal.size)];
            b.doMove(m, st);
            if (!accumulatorMatchesScratch(b, "doMove", game, ply)) return 1;
            ++positions;
        }
    }

    std::cout << "nnue-selftest OK: " << positions << " positions across "
              << games << " games, incremental == scratch everywhere\n";
    return 0;
}

} // namespace NNUE
