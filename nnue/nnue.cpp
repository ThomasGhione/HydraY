#include "nnue.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

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

// Copies a net blob into an owned NetworkDeep (l1w16 and l1wT are derived, so
// the blob cannot be used in place) and makes it active. bullet's format has
// no header: the exact size, padding included, is the first sanity check.
bool activate(const unsigned char* data, size_t size) {
    if (size != (Deep::PAYLOAD_BYTES + 63) / 64 * 64) return false;
    auto deep = std::make_unique<Deep::NetworkDeep>();
    if (!Deep::loadFromMemory(data, size, *deep)) return false;
    ownedDeep = std::move(deep);
    activeNetwork = ownedDeep.get();
    return true;
}

} // namespace

bool loadNetwork(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<unsigned char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!activate(blob.data(), blob.size())) {
        std::cout << "info string EvalFile error: cannot load '" << path << "'\n";
        return false;
    }
    std::cout << "info string EvalFile: network (1024 -> " << Deep::L1_SIZE << " -> 1)"
              << (Deep::hasVnniPath() ? ", VNNI" : ", no VNNI") << "\n";
    return true;
}

bool activateEmbedded() noexcept {
    return activate(g_hydrayEmbeddedNetStart, static_cast<size_t>(g_hydrayEmbeddedNetEnd - g_hydrayEmbeddedNetStart));
}

int32_t evaluate(const chess::Board& b) noexcept {
    // The current frame is computed on demand (board is consistent here).
    b.ensureNnueAccumulatorClean();
    const NNUE::Accumulator& acc = b.accumulator();
    const int stm = chess::Board::colorToIndex(b.getActiveColor()); // 0 = white

    // Material-count output bucket; must match bullet's MaterialCount<8>
    // (and sanity.rs): (popcount - 2) / ceil(32/8). Kings are always on the
    // board, so popcount is in [2, 32] and the bucket in [0, 7].
    const int bucket = (std::popcount(b.getPiecesBitMap()) - 2) / 4;

    return Deep::forwardSimd(*activeNetwork, acc.v[stm], acc.v[stm ^ 1], bucket);
}

} // namespace NNUE
