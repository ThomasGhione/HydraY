#pragma once

// Self-play training-data generation.
// Both entry points are dispatched from main() before Engine is constructed.

namespace NNUE {

// ./chess datagen [--out PREFIX] [--threads N] [--nodes N] [--net FILE]
//                 [--tb DIR] [--shard N] [--seal-on-exit] [--start MODE]
//                 [--eg-every N] [--games N] [--target N] [--seed N]
//                 [--first-shard N] [--machine NAME] [--build ID]
// (./chess datagen --help lists the defaults.) Positions are labeled by an
// NNUE search with the given or embedded net; with tablebases a game is
// adjudicated on entering their range. All threads append bulletformat
// records to <PREFIX>.sNNNN.bin.part, sealed to .bin every --shard positions,
// each with a <PREFIX>.sNNNN.json manifest of its settings, so a run can stop
// (Ctrl+C, SIGTERM) and resume.
int runDatagen(int argc, char* argv[]);

// ./chess datagen-dump <file.bin> [count]
// Prints the record count and the first `count` records as FEN|score|result.
int runDatagenDump(int argc, char* argv[]);

} // namespace NNUE
