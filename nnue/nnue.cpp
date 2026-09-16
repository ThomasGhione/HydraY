#include "nnue.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include "../board/board.hpp"
#include "accumulator.hpp"
#include "network.hpp"
#include "network_deep.hpp"

// Symbols from nnue/embedded.cpp (.incbin of nnue/net/hydray.nnue).
extern "C" const unsigned char g_hydrayEmbeddedNetStart[];
extern "C" const unsigned char g_hydrayEmbeddedNetEnd[];

namespace NNUE {

const Deep::NetworkDeep* activeNetwork = nullptr;

namespace {

std::unique_ptr<Deep::NetworkDeep> ownedDeep;

// File size of a deep net, bullet's padding included.
constexpr size_t DEEP_FILE_BYTES = (Deep::PAYLOAD_BYTES + 63) / 64 * 64;

} // namespace

bool loadNetwork(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cout << "info string EvalFile error: cannot open '" << path << "'\n";
        return false;
    }
    // bullet's format carries no header, so size is the only sanity check
    // available before the loader validates the padding signature.
    if (static_cast<size_t>(in.tellg()) != DEEP_FILE_BYTES) {
        std::cout << "info string EvalFile error: unexpected size (expected "
                  << DEEP_FILE_BYTES << " bytes)\n";
        return false;
    }

    auto deep = std::make_unique<Deep::NetworkDeep>();
    if (!Deep::loadFromFile(path.c_str(), *deep)) {
        std::cout << "info string EvalFile error: failed validation\n";
        return false;
    }
    ownedDeep = std::move(deep);
    activeNetwork = ownedDeep.get();
    std::cout << "info string EvalFile: network (1024 -> " << Deep::L1_SIZE << " -> 1)"
              << (Deep::hasVnniPath() ? ", VNNI" : ", no VNNI") << "\n";
    return true;
}

size_t embeddedSize() noexcept {
    return static_cast<size_t>(g_hydrayEmbeddedNetEnd - g_hydrayEmbeddedNetStart);
}

bool activateEmbedded() noexcept {
    // The net cannot be a plain overlay on the blob -- l1w16 and l1wT are
    // derived, not stored -- so it is copied into an owned NetworkDeep.
    if (embeddedSize() != DEEP_FILE_BYTES) return false;
    auto deep = std::make_unique<Deep::NetworkDeep>();
    if (!Deep::loadFromMemory(g_hydrayEmbeddedNetStart, embeddedSize(), *deep)) return false;
    ownedDeep = std::move(deep);
    activeNetwork = ownedDeep.get();
    return true;
}

bool networkLoaded() noexcept {
    return activeNetwork != nullptr;
}

int32_t evaluate(const chess::Board& b) noexcept {
    // HalfKA laziness: settle any perspective dirtied by a king bucket/flip
    // crossing before reading the accumulator (board is consistent here).
    b.ensureNnueAccumulatorClean();
    const NNUE::Accumulator& acc = b.nnueAccumulator;
    const int stm = chess::Board::colorToIndex(b.getActiveColor()); // 0 = white

    // Material-count output bucket; must match bullet's MaterialCount<8>
    // (and sanity.rs): (popcount - 2) / ceil(32/8). Kings are always on the
    // board, so popcount is in [2, 32] and the bucket in [0, 7].
    const int bucket = (std::popcount(b.getPiecesBitMap()) - 2) / 4;

    return Deep::forwardSimd(*activeNetwork, acc.v[stm], acc.v[stm ^ 1], bucket);
}

} // namespace NNUE
