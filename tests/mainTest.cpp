#include "ut.hpp"
#include "../board/board.hpp"
#include "../board/piece.hpp"
#include "../engine/evaluator.hpp"
#include "../engine/perft.hpp"
#include "../engine/search/searcher.hpp"
#include "../nnue/nnue.hpp"

#include <cassert>

namespace {

void testQuiescenceScoresThreefoldAsDrawBeforeStaticEval() {
    // 7 men (output bucket 1): the 3-man version sat in bucket 0, which the
    // v3 net evaluates near 0 (Syzygy adjudication starves it of endgames).
    chess::Board board("k7/pp6/8/8/8/8/PP6/K6R w - - 0 1");

    assert(board.move({chess::parseSquare("h1"), chess::parseSquare("g1")}));
    assert(board.move({chess::parseSquare("a8"), chess::parseSquare("b8")}));
    assert(board.move({chess::parseSquare("g1"), chess::parseSquare("h1")}));
    assert(board.move({chess::parseSquare("b8"), chess::parseSquare("a8")}));
    assert(board.move({chess::parseSquare("h1"), chess::parseSquare("g1")}));
    assert(board.move({chess::parseSquare("a8"), chess::parseSquare("b8")}));
    assert(board.move({chess::parseSquare("g1"), chess::parseSquare("h1")}));
    assert(board.move({chess::parseSquare("b8"), chess::parseSquare("a8")}));

    assert(board.isThreefoldRepetition());
    assert(engine::Evaluator::evaluate(board) > 0);

    engine::Searcher::SearchRuntime runtime{};
    const int32_t qScore = engine::Searcher::quiescenceSearch(
        board,
        runtime,
        engine::Searcher::NEG_INF,
        engine::Searcher::POS_INF,
        0,
        nullptr,
        false);

    assert(qScore < 0);
}

void testRootDrawTerminalDoesNotSearchFallbackMove() {
    chess::Board board("k7/8/8/8/8/8/8/KB6 w - - 0 1");
    engine::Searcher::SearchRuntime runtime{};

    assert(board.isDraw(board.getActiveColor()));
    const chess::Move bestMove = engine::Searcher::searchBestMove(board, runtime, 4);

    // A terminal-draw root returns a legal fallback move without searching.
    assert(chess::isValidSquare(bestMove.from));
    assert(chess::isValidSquare(bestMove.to));
    assert(runtime.eval == 0);
    assert(runtime.nodesSearched == 0);
}

void testBoardMoveRejectsWrongSideToMove() {
    chess::Board board;
    assert(!board.move({chess::parseSquare("e7"), chess::parseSquare("e5")}));
    assert(board.getActiveColor() == chess::Board::WHITE);
    assert(board.move({chess::parseSquare("e2"), chess::parseSquare("e4")}));
    assert(!board.move({chess::parseSquare("d2"), chess::parseSquare("d4")}));
}

// Exhaustive move-generation check: the standard perft positions against their
// published node counts. Depth 4 is ~11M nodes (well under a second) and
// already covers castling, en passant, promotions, pins and evasions; deeper
// runs are available on demand via `./chess perft suite <depth>`.
void testPerftStandardPositions() {
    // Called outside the assert so the check survives a -DNDEBUG build.
    const bool passed = engine::runPerftSuite(4, /*verbose=*/false);
    assert(passed);
    (void)passed;
}

} // namespace

int main(){
    // Magic bitboards + embedded NNUE net: tests drive Board/Searcher directly
    // (no Engine), so nobody else activates the evaluator for them.
    pieces::initMagicBitboards();
    if (!NNUE::activateEmbedded()) return 1;
    testPerftStandardPositions();
    testQuiescenceScoresThreefoldAsDrawBeforeStaticEval();
    testRootDrawTerminalDoesNotSearchFallbackMove();
    testBoardMoveRejectsWrongSideToMove();
}
