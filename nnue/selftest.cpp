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

bool accumulatorMatchesScratch(const chess::Board& b, const char* context, int game, int ply) {
    b.ensureNnueAccumulatorClean();
    const chess::Board scratch(b.boardToFen());   // its frame is a from-scratch rebuild
    const Accumulator& inc = b.accumulator();
    const Accumulator& ref = scratch.accumulator();
    const bool same = std::memcmp(inc.v, ref.v, sizeof(inc.v)) == 0
                   && inc.base[0] == ref.base[0] && inc.base[1] == ref.base[1]
                   && inc.flip[0] == ref.flip[0] && inc.flip[1] == ref.flip[1]
                   && !inc.dirty[0] && !inc.dirty[1];
    if (!same) {
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

    // Random walks. At every ply a probe move is played and evaluated one ply
    // deeper, then unwound, and the main line is checked only every few plies,
    // so frames are computed from ancestors several moves back, as in a search.
    // Covers captures/EP/castling/promotions and king bucket crossings; every
    // check compares with a board built from scratch.
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
            const MoveList replies = engine::MoveGenerator::generateLegalMoves(b);
            if (!replies.is_empty()) {
                chess::Board::MoveState replyState{};
                const chess::Move& reply = replies[static_cast<size_t>(ply) % static_cast<size_t>(replies.size)];
                b.doMove(reply, replyState);
                if (!accumulatorMatchesScratch(b, "probe reply", game, ply)) return 1;
                b.undoMove(reply, replyState);
            }
            if (ply % 2 == 0 && !accumulatorMatchesScratch(b, "probe", game, ply)) return 1;
            b.undoMove(probe, st);
            if (ply % 3 == 0 && !accumulatorMatchesScratch(b, "do+undo", game, ply)) return 1;

            const chess::Move& m = legal[rng() % static_cast<uint64_t>(legal.size)];
            b.doMove(m, st);
            if (ply % 4 == 3 && !accumulatorMatchesScratch(b, "doMove", game, ply)) return 1;
            ++positions;
        }
    }

    std::cout << "nnue-selftest OK: " << positions << " positions across "
              << games << " games, incremental == scratch everywhere\n";
    return 0;
}

} // namespace NNUE
