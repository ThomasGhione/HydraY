#include "../board.hpp"

#include "../../tests/ut.hpp"
namespace ut = boost::ut;
using namespace ut;

// A plain lambda, registered from main: ut's cfg is a variable template, whose
// dynamic initialization is unordered, and clang builds it after a global
// ut::suite has registered, which drops the suite.
const auto boardSuite = [] {
  "fen_to_board"_test = []{
    // a8..h1, the FEN's own order; '.' is an empty square.
    const std::string_view expected =
        "r.bqkbnr" "pp.ppppp" "..n....." "..p....." "....P..." ".....N.." "PPPP.PPP" "RNBQKB.R";
    const chess::Board b("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    for (uint8_t sq = 0; sq < 64; ++sq) {
      const char c = expected[sq];
      const size_t piece = std::string_view(".PNBRQK").find(static_cast<char>(std::toupper(c)))
                         | (std::isupper(c) ? chess::Board::WHITE : chess::Board::BLACK);
      expect(b.get(sq) == piece) << "square " << int(sq) << "\n";
    }
  };

  "fen_round_trips"_test = []{
    expect(chess::Board().boardToFen() == chess::Board::STARTING_FEN);
    for (const char* fen : {
           "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
           "r3kb1r/pp1n1ppp/1qpp4/8/2P2B2/2Q2BP1/PP2P2P/R3K2R b KQkq - 5 15",
           "r3kb1r/pp1n1ppp/2p1bq2/8/2Pp4/5PP1/PP1BP1BP/2RQK1NR b Kkq - 4 11",
           "r2qk2r/1bpnbpp1/pp1p3p/3Pp3/P1B1P3/2N1nN2/1PPQ1PPP/R4RK1 w kq - 0 12",
           "r4rk1/1b3p1p/p2b1p2/1pp5/8/1BP3N1/PP3PPP/R4RK1 w - - 0 18",
           "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1"}) {
      expect(chess::Board(fen).boardToFen() == fen) << fen << "\n";
    }
  };

  "knight_check_on_c2_is_not_false_mate"_test = []{
    chess::Board b{};
    b.fenToBoard("r1b1kb1r/ppp1qppp/8/8/4Q3/8/PPnN1PPP/R1B1KB1R w KQkq - 0 2");
    const chess::Square kingFrom = chess::parseSquare("e1");
    const chess::Square escapeSq = chess::parseSquare("d1");

    expect(b.inCheck(chess::Board::WHITE));

    expect(b.isLegalPseudoMove(
      kingFrom,
      escapeSq,
      b.get(kingFrom)
    )) << "White can answer the knight check with Kd1\n";
  };

  "repetition_inside_null_move_subtree"_test = []{
    // A cycle played entirely after a null move is a real repetition of the
    // post-null position and must be counted.
    chess::Board b{};
    b.fenToBoard("4k3/8/8/8/8/8/8/1N2K1n1 w - - 10 40");
    const auto mv = [](const char* uci) {
      return chess::Move{chess::parseSquare(std::string_view(uci, 2)),
                         chess::parseSquare(std::string_view(uci + 2, 2)), 0};
    };
    const char* cycle[4] = {"g1f3", "b1c3", "f3g1", "c3b1"};
    chess::Board::MoveState nullState{};
    chess::Board::MoveState st[8]{};
    b.doNullMove(nullState);
    expect(b.countRepetitions() == 1);
    for (int i = 0; i < 4; ++i) b.doMove(mv(cycle[i]), st[i]);
    expect(b.countRepetitions() == 2) << "second occurrence of the post-null position\n";
    for (int i = 0; i < 4; ++i) b.doMove(mv(cycle[i]), st[4 + i]);
    expect(b.countRepetitions() == 3) << "third occurrence of the post-null position\n";
    for (int i = 7; i >= 0; --i) b.undoMove(mv(cycle[i % 4]), st[i]);
    b.undoNullMove(nullState);
    expect(b.countRepetitions() == 1);
  };

};

void registerBoardSuite() { [[maybe_unused]] const ut::suite registered = boardSuite; }
