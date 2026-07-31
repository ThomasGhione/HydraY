// Not cover all functions

#include "../board.hpp"

#include "../../tests/ut.hpp"
namespace ut = boost::ut;
using namespace ut;

constexpr uint8_t wpawn = ( (chess::Board::WHITE) | (chess::Board::PAWN));
constexpr uint8_t bpawn = ( (chess::Board::BLACK) | (chess::Board::PAWN));

constexpr uint8_t wrook = ( (chess::Board::WHITE) | (chess::Board::ROOK));
constexpr uint8_t brook = ( (chess::Board::BLACK) | (chess::Board::ROOK));

constexpr uint8_t wknight = ( (chess::Board::WHITE) | (chess::Board::KNIGHT));
constexpr uint8_t bknight = ( (chess::Board::BLACK) | (chess::Board::KNIGHT));

constexpr uint8_t wbishop = ( (chess::Board::WHITE) | (chess::Board::BISHOP));
constexpr uint8_t bbishop = ( (chess::Board::BLACK) | (chess::Board::BISHOP));

constexpr uint8_t wqueen = ( (chess::Board::WHITE) | (chess::Board::QUEEN));
constexpr uint8_t bqueen = ( (chess::Board::BLACK) | (chess::Board::QUEEN));

constexpr uint8_t wking = ( (chess::Board::WHITE) | (chess::Board::KING));
constexpr uint8_t bking = ( (chess::Board::BLACK) | (chess::Board::KING));

constexpr uint8_t empty = (chess::Board::EMPTY) ;

void controlExpect(const chess::Board b, std::array<uint8_t, 64> expectPos){
  const char coll[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  
  int indexArray = 0;
  for(int i = 1; i <= 8; i++){
    for(char colll : coll){
      expect(b.get(chess::parseSquare(colll + std::to_string(i))) == expectPos.at(indexArray));

      indexArray++;
    }
  }
}

ut::suite boardSuite = [] {
  "Default constructor"_test = []{
    chess::Board b = chess::Board();
  
    
    expect(b.get(0,0) == (wrook) );
  };

  "fen_to_board-1"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";

    b.fenToBoard(FEN_TEST);

    std::array<uint8_t, 64> expectPos = {wrook, wknight, wbishop, wqueen, wking, wbishop, empty, wrook,
                                        wpawn, wpawn, wpawn, wpawn, empty, wpawn, wpawn, wpawn,
                                        empty, empty, empty, empty, empty, wknight, empty, empty,
                                        empty, empty, empty, empty, wpawn, empty, empty, empty,
                                        empty, empty, bpawn, empty, empty, empty, empty, empty,
                                        empty, empty, bknight, empty, empty, empty, empty, empty,
                                        bpawn, bpawn, empty, bpawn, bpawn, bpawn, bpawn, bpawn,
                                        brook, empty, bbishop, bqueen, bking, bbishop, bknight, brook};

    controlExpect(b, expectPos);
  };

  "fen_to_board-2"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r3kb1r/pp1n1ppp/1qpp4/8/2P2B2/2Q2BP1/PP2P2P/R3K2R b KQkq - 5 15";

    b.fenToBoard(FEN_TEST);
    
    std::array<uint8_t, 64> expectPos = {wrook, empty, empty, empty, wking, empty, empty, wrook,
                              wpawn, wpawn, empty, empty, wpawn, empty, empty, wpawn,
                              empty, empty, wqueen, empty, empty, wbishop, wpawn, empty,
                              empty, empty, wpawn, empty, empty, wbishop, empty, empty,
                              empty, empty, empty, empty, empty, empty, empty, empty,
                              empty, bqueen, bpawn, bpawn, empty, empty, empty, empty,
                              bpawn, bpawn, empty, bknight, empty, bpawn, bpawn, bpawn,
                              brook, empty, empty, empty, bking, bbishop, empty, brook};

    controlExpect(b, expectPos);
  };
  
  "fen_to_board-3"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r3kb1r/pp1n1ppp/2p1bq2/8/2Pp4/5PP1/PP1BP1BP/2RQK1NR b Kkq - 4 11";

    b.fenToBoard(FEN_TEST);
    
    std::array<uint8_t, 64> expectPos = {empty, empty, wrook, wqueen, wking, empty, wknight, wrook,
                              wpawn, wpawn, empty, wbishop, wpawn, empty, wbishop, wpawn,
                              empty, empty, empty, empty, empty, wpawn, wpawn, empty,
                              empty, empty, wpawn, bpawn, empty, empty, empty, empty,
                              empty, empty, empty, empty, empty, empty, empty, empty,
                              empty, empty, bpawn, empty, bbishop, bqueen, empty, empty,
                              bpawn, bpawn, empty, bknight, empty, bpawn, bpawn, bpawn,
                              brook, empty, empty, empty, bking, bbishop, empty, brook};

    controlExpect(b, expectPos);
  };
  
  "fen_to_board-4"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r2qk2r/1bpnbpp1/pp1p3p/3Pp3/P1B1P3/2N1nN2/1PPQ1PPP/R4RK1 w kq - 0 12";

    b.fenToBoard(FEN_TEST);

    std::array<uint8_t, 64> expectPos = {wrook, empty, empty, empty, empty, wrook, wking, empty,
                                     empty, wpawn, wpawn, wqueen, empty, wpawn, wpawn, wpawn,
                                     empty, empty, wknight, empty, bknight, wknight, empty, empty,
                                     wpawn, empty, wbishop, empty, wpawn, empty, empty, empty,
                                     empty, empty, empty, wpawn, bpawn, empty, empty, empty,
                                     bpawn, bpawn, empty, bpawn, empty, empty, empty, bpawn,
                                     empty, bbishop, bpawn, bknight, bbishop, bpawn, bpawn, empty,
                                     brook, empty, empty, bqueen, bking, empty, empty, brook};

    controlExpect(b, expectPos);
  };
  
  "fen_to_board-5"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r4rk1/1b3p1p/p2b1p2/1pp5/8/1BP3N1/PP3PPP/R4RK1 w - - 0 18";

    b.fenToBoard(FEN_TEST);

    std::array<uint8_t, 64> expectPos = {wrook, empty, empty, empty, empty, wrook, wking, empty,
                                    wpawn, wpawn, empty, empty, empty, wpawn, wpawn, wpawn,
                                    empty, wbishop, wpawn, empty, empty, empty, wknight, empty,
                                    empty, empty, empty, empty, empty, empty, empty, empty,
                                    empty, bpawn, bpawn, empty, empty, empty, empty, empty,
                                    bpawn, empty, empty, bbishop, empty, bpawn, empty, empty,
                                    empty, bbishop, empty, empty, empty, bpawn, empty, bpawn,
                                    brook, empty, empty, empty, empty, brook, bking, empty};

    controlExpect(b, expectPos);
  };
  "fen_to_board-6"_test = []{
    chess::Board b = chess::Board();
    
    const std::string FEN_TEST = "r3kb1r/pp1n1ppp/2p1bq2/8/2Pp4/5PP1/PP1BP1BP/2RQK1NR b Kkq - 4 11";

    b.fenToBoard(FEN_TEST);
    
    std::array<uint8_t, 64> expectPos = { empty, empty, wrook, wqueen, wking, empty, wknight, wrook,
                                    wpawn, wpawn, empty, wbishop, wpawn, empty, wbishop, wpawn,
                                    empty, empty, empty, empty, empty, wpawn, wpawn, empty,
                                    empty, empty, wpawn, bpawn, empty, empty, empty, empty,
                                    empty, empty, empty, empty, empty, empty, empty, empty,
                                    empty, empty, bpawn, empty, bbishop, bqueen, empty, empty,
                                    bpawn, bpawn, empty, bknight, empty, bpawn, bpawn, bpawn,
                                    brook, empty, empty, empty, bking, bbishop, empty, brook};

    controlExpect(b, expectPos);
  };

  "boardToFen_starting_position"_test = []{
    chess::Board b{}; // default ctor loads the starting FEN
    const std::string expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    expect(b.boardToFen() == expected) 
      << "Current fen: " << b.boardToFen() << "\n" 
      << "expected fen: " << expected << "\n";
  };
  
  "boardToFen_roundtrip_examples_1"_test = []{
    chess::Board b{};
    const std::string expected = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
    b.fenToBoard(expected);
    expect(b.boardToFen() == expected) 
      << "Current fen: " << b.boardToFen() << "\n" 
      << "expected fen: " << expected << "\n";
  };
  
  "boardToFen_roundtrip_examples_2"_test = []{
    chess::Board b{};
    const std::string expected = "r3kb1r/pp1n1ppp/1qpp4/8/2P2B2/2Q2BP1/PP2P2P/R3K2R b KQkq - 5 15";
    b.fenToBoard(expected);
    expect(b.boardToFen() == expected);
  };


  "boardToFen_roundtrip_with_enpassant"_test = []{
    chess::Board b{};
    // Includes en passant square and maintains active color and clocks
    const std::string expected = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1";
    b.fenToBoard(expected);
    expect(b.boardToFen() == expected);
  };

  "knight_check_on_c2_is_not_false_mate"_test = []{
    chess::Board b{};
    b.fenToBoard("r1b1kb1r/ppp1qppp/8/8/4Q3/8/PPnN1PPP/R1B1KB1R w KQkq - 0 2");
    const chess::Square kingFrom = chess::parseSquare("e1");
    const chess::Square escapeSq = chess::parseSquare("d1");

    expect(b.inCheck(chess::Board::WHITE));
    expect(b.hasAnyLegalMove(chess::Board::WHITE));
    expect(!b.isCheckmate(chess::Board::WHITE));

    expect(b.isLegalPseudoMove(
      kingFrom,
      escapeSq,
      b.get(kingFrom)
    )) << "White can answer the knight check with Kd1\n";
  };

};
