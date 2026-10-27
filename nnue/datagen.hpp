#pragma once

// Self-play training-data generation.
// Both entry points are dispatched from main() before Engine is constructed.

namespace NNUE {

// ./chess datagen [outPrefix] [threads] [nodesPerMove] [netPath] [tbPath]
//   defaults: nnue/data/hydray  3  8000  <embedded net>  engine/syzygy/files
// Positions are labeled by an NNUE search with the given or embedded net; with
// tablebases a game is adjudicated on entering their range. Each thread
// appends bulletformat records to <outPrefix>.t<N>.bin, so a run can stop
// (Ctrl+C, SIGTERM) and resume.
int runDatagen(int argc, char* argv[]);

// ./chess datagen-dump <file.bin> [count]
// Prints the record count and the first `count` records as FEN|score|result.
int runDatagenDump(int argc, char* argv[]);

} // namespace NNUE
