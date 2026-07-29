// Performance tests for the chess engine

#include <bit>
#include "../../engine.hpp"
#include "../../evaluator.hpp"
#include "../../../nnue/nnue.hpp"
#include "../../sort/move_generator.hpp"
#include "../../movelist.hpp"
#include "../../../tests/ut.hpp"

namespace ut = boost::ut;

ut::suite performanceEngineSuite = [] {
  using namespace ut;

  "performance searchPosition depth 11"_test = []{
    engine::Engine e = engine::Engine();
    constexpr int depth = 11;

    auto start = std::chrono::high_resolution_clock::now();
    e.searchUCI(engine::time::Limits{.maxDepth = depth});
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Depth 11 search completed in %lu ms\n", duration);
    expect(duration < 9000);
  };

    
  "avg performance searchPosition depth 10 over 20 runs"_test = []{
    engine::Engine e = engine::Engine();
    constexpr int depth = 10;

    constexpr int runs = 20;
    int64_t totalDuration = 0;

    // Plays against itself for "runs" moves. searchUCI() searches a COPY of the
    // board and does NOT advance it, so we must play the returned move on e.board
    // ourselves — otherwise every run re-searches the identical root position and
    // hits a fully warm TT (~0 ms at any depth), measuring nothing.
    int completedRuns = 0;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const chess::Move move = e.searchUCI(engine::time::Limits{.maxDepth = depth});
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Run %d completed in %lu ms\n", i + 1, duration);
        totalDuration += duration;
        ++completedRuns;

        // Advance the self-play game so the next run is a new position; stop if
        // there is no legal move (checkmate / stalemate / terminal).
        if (!chess::isValidSquare(move.from) || !chess::isValidSquare(move.to)) break;
        chess::Board::MoveState state;
        e.board.doMove(move, state);
    }

    double avgDuration = static_cast<double>(totalDuration) / completedRuns;

    // Attesa che la ricerca media venga completata in meno di 500 millisecondi
    printf("Average Depth 10 search time over %d runs: %.2f ms\n\n", runs, avgDuration);
    expect(avgDuration < 700);
  };

  "benchmark evaluate (NNUE) and movegen"_test = []{
    constexpr std::array<const char*, 16> ALL_FENS = {
      // 8 normal
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3kbnr/pppbpppp/4q3/8/1n6/P1NPB3/1PP1NPPP/R2QKB1R b KQkq - 0 1",
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
      "rnbqk2r/ppp2ppp/4pn2/3p4/1b1P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
      "r2q1rk1/pp2bppp/2np1n2/2p1p3/2P1P3/2NP1N2/PP2BPPP/R1BQ1RK1 w - - 0 9",
      "8/2p5/3p2k1/2P1p1p1/4P3/3P1K2/8/8 w - - 0 40",
      "4rrk1/pp3ppp/2n1bn2/2bp4/3P4/2P1PN2/PP1NBPPP/R2QR1K1 w - - 4 13",
      "r3k2r/pppq1ppp/2npbn2/3Np3/2P1P3/2N1B3/PP2QPPP/R3KB1R w KQkq - 2 10",
      // 8 endgame
      "8/p7/4k3/1R6/2K5/8/P7/8 w - - 0 1",
      "8/8/8/3R4/8/3k4/8/1R3K2 w - - 0 1",
      "8/8/1r6/8/8/3k4/8/1R3K2 w - - 0 1",
      "8/8/8/3R4/8/3k4/8/1R1R1K2 w - - 0 1",
      "r1b1k2r/pppq1ppp/2npbn2/3Np3/2P1P3/2N1B3/PP2QPPP/R3KB1R w KQkq - 2 10",
      "rnbqk2r/ppp2ppp/4pn2/3p4/1b1P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
      "r2q1rk1/pp2bppp/2np1n2/2p1p3/2P1P3/2NP1N2/PP2BPPP/R1BQ1RK1 w - - 0 9",
      "8/2p5/3p2k1/2P1p1p1/4P3/3P1K2/8/8 w - - 0 40"
    };

    // Boards must be built AFTER the network is active so their accumulators
    // are filled during FEN load (Evaluator::evaluate is NNUE-only).
    expect(NNUE::activateEmbedded());
    std::array<chess::Board, 16> boards;
    for (int i = 0; i < 16; ++i) {
      boards[i] = chess::Board(ALL_FENS[i]);
    }

    constexpr int EVAL_ITERATIONS = 500'000;
    int64_t evalSink = 0;
    auto evalStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < EVAL_ITERATIONS; ++iter) {
      for (int i = 0; i < 16; ++i) {
        evalSink += engine::Evaluator::evaluate(boards[i]);
      }
    }
    auto evalEnd = std::chrono::high_resolution_clock::now();
    auto evalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(evalEnd - evalStart).count();
    printf("NNUE evaluate on 16 FENs * %d iterations took %lu ms\n", EVAL_ITERATIONS, static_cast<unsigned long>(evalDuration));
    printf("Eval sink: %lld\n\n", static_cast<long long>(evalSink));

    constexpr int ITERATIONS = 500'000;
    int64_t dummySink = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      for (int i = 0; i < 16; ++i) {
        MoveList moves = engine::MoveGenerator::generateLegalMoves(boards[i]);
        dummySink += moves.size;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("generateLegalMoves on 16 FENs * %d iterations took %lu ms\n", ITERATIONS, static_cast<unsigned long>(duration));
    printf("Dummy sink: %lld\n\n", static_cast<long long>(dummySink));

    expect(true);
    printf("\nBenchmark completed.\n\n");
  };
};
