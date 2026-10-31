#include "nnue.hpp"

#include <bit>
#include <charconv>
#include <cstdlib>
#include <string_view>
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

// Symbols from nnue/embedded.cpp (.incbin of nnue/net/hydray.nnue and nnue/scale_ref.txt).
extern "C" const unsigned char g_hydrayEmbeddedNetStart[];
extern "C" const unsigned char g_hydrayEmbeddedNetEnd[];
extern "C" const unsigned char g_hydrayScaleRefStart[];
extern "C" const unsigned char g_hydrayScaleRefEnd[];

namespace NNUE {

const Deep::NetworkDeep* activeNetwork = nullptr;

namespace {

std::unique_ptr<Deep::NetworkDeep> ownedDeep;
double scale = 1.0;

// Eval normalisation. Training lets the output scale drift from net to net
// (about +12% from 4.1.0 to 4.2.0), and every margin, adjudication threshold
// and label is in eval units. scale_ref.txt holds 2048 quiet positions with
// the 4.2.0 net's static eval, |eval| in [100, 1500]: each net is scaled so
// that the sum of absolute evals over them matches. The 4.2.0 net gets 1.0
// exactly, so its evals are unchanged to the bit.
double measureScale() {
    const std::string_view text(reinterpret_cast<const char*>(g_hydrayScaleRefStart),
                                static_cast<size_t>(g_hydrayScaleRefEnd - g_hydrayScaleRefStart));
    chess::Board b;
    int64_t refSum = 0, netSum = 0;
    for (size_t pos = 0; pos < text.size();) {
        size_t end = text.find('\n', pos);
        if (end == std::string_view::npos) end = text.size();
        const std::string_view line = text.substr(pos, end - pos);
        pos = end + 1;
        if (line.empty() || line.front() == '#') continue;
        const size_t space = line.find(' ');
        int ref = 0;
        std::from_chars(line.data(), line.data() + space, ref);
        b.fenToBoard(std::string(line.substr(space + 1)));
        refSum += std::abs(ref);
        netSum += std::abs(evaluate(b));
    }
    return netSum > 0 ? static_cast<double>(refSum) / static_cast<double>(netSum) : 1.0;
}

// Copies a net blob into an owned NetworkDeep (l1w16 and l1wT are derived, so
// the blob cannot be used in place), makes it active and normalises it.
// bullet's format has no header: the exact size, padding included, is the
// first sanity check, and loadFromMemory makes it.
bool activate(const unsigned char* data, size_t size) {
    auto deep = std::make_unique<Deep::NetworkDeep>();
    if (!Deep::loadFromMemory(data, size, *deep)) return false;
    ownedDeep = std::move(deep);
    activeNetwork = ownedDeep.get();
    scale = measureScale();
    ownedDeep->outputScale = static_cast<float>(Deep::SCALE * scale);
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
    std::cout << "info string EvalFile: network (1024 -> " << Deep::L1_SIZE << " -> 1, "
              << activeNetwork->fileBuckets << " output buckets)"
              << (Deep::hasVnniPath() ? ", VNNI" : ", no VNNI") << ", eval scale x" << scale << "\n";
    return true;
}

double evalScale() noexcept { return scale; }

bool activateEmbedded() noexcept {
    return activate(g_hydrayEmbeddedNetStart, static_cast<size_t>(g_hydrayEmbeddedNetEnd - g_hydrayEmbeddedNetStart));
}

int32_t evaluate(const chess::Board& b) noexcept {
    // The current frame is computed on demand (board is consistent here).
    b.ensureNnueAccumulatorClean();
    const NNUE::Accumulator& acc = b.accumulator();
    const int stm = chess::Board::colorToIndex(b.getActiveColor()); // 0 = white

    // Material-count output bucket; must match bullet's MaterialCount<OUTPUT_BUCKETS>
    // (and sanity_deep.rs). Kings are always on the board, so popcount is in
    // [2, 32] and the bucket in [0, OUTPUT_BUCKETS).
    const int bucket = (std::popcount(b.getPiecesBitMap()) - 2) / Deep::BUCKET_DIVISOR;

    return Deep::forwardSimd(*activeNetwork, acc.v[stm], acc.v[stm ^ 1], bucket);
}

} // namespace NNUE
