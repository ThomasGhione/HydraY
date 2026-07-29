#include "datagen.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../board/board.hpp"
#include "../board/piece.hpp"
#include "../engine/search/searcher.hpp"
#include "../engine/sort/move_generator.hpp"
#include "../engine/syzygy/syzygy.hpp"
#include "../tt/tt.hpp"
#include "bulletformat.hpp"
#include "nnue.hpp"

namespace NNUE {

namespace {

// Defaults and filters per NNUE_PLAN.md (Fase 0). The bestmove-is-tactical
// skip is the one addition: a position whose search score hinges on an
// unresolved capture/promotion is a noisy static-eval target.
constexpr int      DEFAULT_THREADS          = 3;
constexpr uint64_t DEFAULT_NODES_PER_MOVE   = 8000;
constexpr int      TARGET_DEPTH_CAP         = 32;
constexpr int      OPENING_PLIES_MIN        = 8;   // +1 at random -> 8 or 9
constexpr int32_t  OPENING_MAX_IMBALANCE_CP = 400;
constexpr int      MIN_RECORD_PLY           = 16;
constexpr int32_t  MAX_RECORD_SCORE_CP      = 3000;
constexpr int32_t  WIN_ADJ_CP               = 2500;
constexpr int      WIN_ADJ_PLIES            = 4;
constexpr int32_t  DRAW_ADJ_CP              = 10;
constexpr int      DRAW_ADJ_PLIES           = 8;
constexpr int      DRAW_ADJ_MIN_PLY         = 80;
constexpr int      MAX_GAME_PLIES           = 400;
constexpr uint64_t TARGET_POSITIONS         = 5'000'000'000; // v4 target (ETA line only)
// WDL probes ignore the 50-move counter: with a high clock a "won" table
// position can still be drawn by the rule before conversion. Draw results
// are always safe; decisive ones are trusted only below this clock.
constexpr int      DECISIVE_TB_ADJ_MAX_HMC  = 60;

std::atomic<bool>     g_stop{false};
std::atomic<uint64_t> g_totalPositions{0};
std::atomic<uint64_t> g_totalGames{0};
std::atomic<uint64_t> g_totalTbAdjudications{0};

// Shared, read-only after load; pyrrhic WDL probes are thread-safe post-init
// (the search already probes from concurrent Lazy SMP threads).
syzygy::SyzygyProber g_syzygy;

void onStopSignal(int) { g_stop.store(true, std::memory_order_release); }

// Everything one worker owns; heap-allocated (SearchRuntime alone is ~1.7 MB).
struct WorkerContext {
    engine::SearchRuntime runtime;
    TT tt;
    std::atomic<bool> interrupted{false};
    std::mt19937_64 rng;
    std::ofstream out;

    WorkerContext(uint64_t seed, const std::string& outPath)
        : rng(seed), out(outPath, std::ios::binary | std::ios::app) {
        runtime.transpositionTable = &tt;
        // Without this binding the node-cap abort in enterNode() is invisible
        // to runIterativeDeepening and truncated iterations would be trusted.
        runtime.searchInterrupted = &interrupted;
        runtime.stopSearchRequested = &g_stop;
    }
};

// Plays one self-play game; returns the number of positions written
// (0 = game discarded: unbalanced opening, dead-end opening, or stop request).
uint64_t playOneGame(WorkerContext& w, uint64_t nodesPerMove) {
    using chess::Board;

    Board b{};
    const int openingPlies = OPENING_PLIES_MIN + static_cast<int>(w.rng() & 1);
    for (int i = 0; i < openingPlies; ++i) {
        const MoveList ml = engine::MoveGenerator::generateLegalMoves(b);
        if (ml.is_empty()) return 0;
        Board::MoveState st{};
        b.doMove(ml[w.rng() % static_cast<uint64_t>(ml.size)], st);
    }

    // Same decay a fresh `go` gets: keeps cross-game history learning mild.
    w.runtime.softResetHistory();

    std::vector<PendingRecord> pending;
    pending.reserve(128);
    int whiteWinStreak = 0;
    int blackWinStreak = 0;
    int drawStreak = 0;
    int whiteResultX2 = -1;
    int ply = openingPlies;

    while (true) {
        if (g_stop.load(std::memory_order_acquire)) return 0;

        const uint8_t active = b.getActiveColor();
        const MoveList legal = engine::MoveGenerator::generateLegalMoves(b);
        if (legal.is_empty()) {
            whiteResultX2 = b.inCheck(active) ? (active == Board::WHITE ? 0 : 2) : 1;
            break;
        }
        if (b.isFiftyMoveRule() || b.isThreefoldRepetition()
            || b.hasInsufficientMaterialDraw() || ply >= MAX_GAME_PLIES) {
            whiteResultX2 = 1;
            break;
        }

        // Syzygy adjudication: as soon as the position enters TB range, close
        // the game with the exact tablebase result — every record collected so
        // far gets a perfect outcome label and the game ends sooner. Tables
        // assume no castling rights (guard below); cursed wins / blessed
        // losses are draws under the 50-move rule and map to draw here.
        if (g_syzygy.isLoaded() && g_syzygy.inTBRange(b)
            && !(b.getCastle(0) || b.getCastle(1) || b.getCastle(2) || b.getCastle(3))) {
            if (const auto wdl = g_syzygy.probeWDL(b)) {
                const bool stmWin  = (*wdl == syzygy::WDL::Win);
                const bool stmLoss = (*wdl == syzygy::WDL::Loss);
                if (!stmWin && !stmLoss) {
                    whiteResultX2 = 1;
                    g_totalTbAdjudications.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                if (b.getHalfMoveClock() <= DECISIVE_TB_ADJ_MAX_HMC) {
                    const int stmResultX2 = stmWin ? 2 : 0;
                    whiteResultX2 = (active == Board::WHITE) ? stmResultX2 : 2 - stmResultX2;
                    g_totalTbAdjudications.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                // High 50-move clock on a decisive WDL: play on instead of
                // risking a mislabel.
            }
        }

        w.tt.incrementGeneration();
        w.runtime.nodesSearched = 0;
        w.runtime.maxNodes = nodesPerMove;
        w.runtime.clearInterrupted();
        const auto res =
            engine::Searcher::runIterativeDeepening(b, w.runtime, 1, TARGET_DEPTH_CAP);

        // Depth 1 not finishing within the node budget is pathological but
        // possible; play the first legal move and record nothing.
        const bool searchOk = res.completedAnyDepth && !(res.bestMove == chess::Move{});
        const chess::Move best = searchOk ? res.bestMove : legal[0];
        const int32_t stmScore = res.bestScore;
        const int32_t whiteScore = (active == Board::WHITE) ? stmScore : -stmScore;

        if (ply == openingPlies && std::abs(whiteScore) > OPENING_MAX_IMBALANCE_CP) {
            return 0;
        }

        const bool isCapture = (b.get(best.to) != Board::EMPTY)
            || ((b.get(best.from) & Board::MASK_PIECE_TYPE) == Board::PAWN
                && best.to == b.getEnPassant());
        const bool isTactical = isCapture || best.promotionType != 0;
        if (searchOk && ply >= MIN_RECORD_PLY && !isTactical && !b.inCheck(active)
            && std::abs(whiteScore) <= MAX_RECORD_SCORE_CP) {
            pending.push_back(packPosition(b, static_cast<int16_t>(whiteScore)));
        }

        whiteWinStreak = (whiteScore >=  WIN_ADJ_CP) ? whiteWinStreak + 1 : 0;
        blackWinStreak = (whiteScore <= -WIN_ADJ_CP) ? blackWinStreak + 1 : 0;
        if (whiteWinStreak >= WIN_ADJ_PLIES) { whiteResultX2 = 2; break; }
        if (blackWinStreak >= WIN_ADJ_PLIES) { whiteResultX2 = 0; break; }
        drawStreak = (ply >= DRAW_ADJ_MIN_PLY && std::abs(whiteScore) <= DRAW_ADJ_CP)
            ? drawStreak + 1 : 0;
        if (drawStreak >= DRAW_ADJ_PLIES) { whiteResultX2 = 1; break; }

        Board::MoveState st{};
        b.doMove(best, st);
        ++ply;
    }

    std::vector<BulletRecord> flat;
    flat.reserve(pending.size());
    for (PendingRecord& p : pending) {
        finalizeResult(p, whiteResultX2);
        flat.push_back(p.record);
    }
    if (!flat.empty()) {
        w.out.write(reinterpret_cast<const char*>(flat.data()),
                    static_cast<std::streamsize>(flat.size() * sizeof(BulletRecord)));
        w.out.flush();
        if (!w.out) {
            std::cerr << "datagen: write failed (disk full?) — stopping.\n";
            g_stop.store(true, std::memory_order_release);
            return 0;
        }
    }

    g_totalPositions.fetch_add(flat.size(), std::memory_order_relaxed);
    g_totalGames.fetch_add(1, std::memory_order_relaxed);
    return flat.size();
}

void workerLoop(int threadId, const std::string& outPath, uint64_t nodesPerMove) {
    const uint64_t seed = std::random_device{}()
        ^ (static_cast<uint64_t>(
               std::chrono::steady_clock::now().time_since_epoch().count())
           << 16)
        ^ (static_cast<uint64_t>(threadId) << 56);
    WorkerContext w(seed, outPath);
    if (!w.out) {
        std::cerr << "datagen: cannot open " << outPath << " for append.\n";
        g_stop.store(true, std::memory_order_release);
        return;
    }
    while (!g_stop.load(std::memory_order_acquire)) {
        playOneGame(w, nodesPerMove);
    }
}

std::string withThreadSuffix(const std::string& prefix, int threadId) {
    return prefix + ".t" + std::to_string(threadId) + ".bin";
}

// Positions already on disk for this prefix (any thread count of past runs):
// the counters resume from the true total instead of restarting at 0.
uint64_t countExistingPositions(const std::string& outPrefix) {
    namespace fs = std::filesystem;
    const fs::path p(outPrefix);
    const fs::path dir = p.parent_path().empty() ? fs::path(".") : p.parent_path();
    const std::string stem = p.filename().string() + ".t";
    uint64_t bytes = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with(stem) && name.ends_with(".bin")) {
            std::error_code sizeEc;
            const uint64_t sz = fs::file_size(entry.path(), sizeEc);
            if (!sizeEc) bytes += sz;
        }
    }
    return bytes / sizeof(BulletRecord);
}

// 1234567 -> "1234K", 464480686 -> "464M": coarse on purpose, the progress
// line is for humans (exact totals stay derivable from the file sizes).
std::string fmtCount(uint64_t n) {
    if (n >= 10'000'000) return std::to_string(n / 1'000'000) + "M";
    if (n >= 10'000)     return std::to_string(n / 1'000) + "K";
    return std::to_string(n);
}

// games/tb-adj are not derivable from the .bin files (positions are), so
// their lifetime totals live in a tiny <prefix>.meta sidecar, rewritten at
// every report tick. Losing it only resets these two display counters.
struct MetaCounters {
    uint64_t games = 0;
    uint64_t tbAdjudications = 0;
};

MetaCounters loadMeta(const std::string& metaPath) {
    MetaCounters m;
    std::ifstream in(metaPath);
    if (in) in >> m.games >> m.tbAdjudications;
    return m;
}

void saveMeta(const std::string& metaPath, const MetaCounters& m) {
    std::ofstream out(metaPath, std::ios::trunc);
    if (out) out << m.games << " " << m.tbAdjudications << "\n";
}

} // namespace

int runDatagen(int argc, char* argv[]) {
    pieces::initMagicBitboards();

    const std::string outPrefix = (argc >= 3) ? argv[2] : "nnue/data/hydray";
    const int threads = (argc >= 4)
        ? std::clamp(std::atoi(argv[3]), 1, 32) : DEFAULT_THREADS;
    const uint64_t nodesPerMove = (argc >= 5)
        ? std::max<uint64_t>(std::strtoull(argv[4], nullptr, 10), 256)
        : DEFAULT_NODES_PER_MOVE;

    // Labeler network: an explicit path when given, the embedded net
    // otherwise. Fail hard on a bad path: silently falling back to a
    // different net would poison days of generation with unintended labels.
    const char* netPath = (argc >= 6) ? argv[5] : nullptr;
    if (netPath != nullptr) {
        if (!loadNetwork(netPath)) {
            std::cerr << "datagen: cannot load network '" << netPath << "'\n";
            return 1;
        }
    } else if (!activateEmbedded()) {
        std::cerr << "datagen: embedded network failed validation\n";
        return 1;
    }

    const std::filesystem::path parent =
        std::filesystem::path(outPrefix).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::cerr << "datagen: cannot create " << parent << ": "
                      << ec.message() << "\n";
            return 1;
        }
    }

    std::signal(SIGINT, onStopSignal);
    std::signal(SIGTERM, onStopSignal);

    // Optional tablebase adjudication: exact result labels + shorter endgames.
    // Missing tables are not an error — endgames are simply played out.
    const char* tbPath = (argc >= 7) ? argv[6] : "engine/syzygy/files";
    const bool tbOn = g_syzygy.load(tbPath) && g_syzygy.maxPieces() >= 3;

    const uint64_t resumedPositions = countExistingPositions(outPrefix);
    const std::string metaPath = outPrefix + ".meta";
    const MetaCounters resumedMeta = loadMeta(metaPath);

    std::cout << "HydraY datagen — bulletformat self-play data\n"
              << "  output : " << outPrefix << ".t<0.." << (threads - 1) << ">.bin (append)\n"
              << "  resume : " << fmtCount(resumedPositions)
              << " positions already on disk for this prefix\n"
              << "  labels : NNUE (" << (netPath != nullptr ? netPath : "embedded") << ")\n"
              << "  threads: " << threads << "  nodes/move: " << nodesPerMove << "\n"
              << "  filters: ply>=" << MIN_RECORD_PLY << ", not in check, quiet bestmove, |cp|<="
              << MAX_RECORD_SCORE_CP << "\n"
              << "  syzygy : " << (tbOn
                     ? std::string("adjudication ON (") + tbPath + ", "
                       + std::to_string(g_syzygy.maxPieces()) + "-man)"
                     : "OFF (no tablebases found - endgames played out)") << "\n"
              << "  stop   : Ctrl+C / SIGTERM (partial games are discarded)\n"
              << std::flush;

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back(workerLoop, t, withThreadSuffix(outPrefix, t), nodesPerMove);
    }

    const auto start = std::chrono::steady_clock::now();
    auto lastReport = start;
    while (!g_stop.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport < std::chrono::seconds(30)) continue;
        lastReport = now;

        const uint64_t generated = g_totalPositions.load(std::memory_order_relaxed);
        const uint64_t total = resumedPositions + generated;
        const double elapsed = std::chrono::duration<double>(now - start).count();
        // Rate (and thus the ETA) reflects THIS run only; the total includes
        // everything already on disk for the prefix.
        const double rate = (elapsed > 0.0) ? static_cast<double>(generated) / elapsed : 0.0;
        const double etaDays = (rate > 0.0)
            ? static_cast<double>(TARGET_POSITIONS - std::min(total, TARGET_POSITIONS))
                / rate / 86400.0
            : 0.0;
        const MetaCounters meta{
            resumedMeta.games + g_totalGames.load(std::memory_order_relaxed),
            resumedMeta.tbAdjudications
                + g_totalTbAdjudications.load(std::memory_order_relaxed)};
        saveMeta(metaPath, meta);
        std::cout << "[datagen] positions " << fmtCount(total)
                  << " (+" << fmtCount(generated) << " run)"
                  << "  games " << fmtCount(meta.games)
                  << "  tb-adj " << fmtCount(meta.tbAdjudications)
                  << "  pos/s " << static_cast<uint64_t>(rate)
                  << "  ETA(5B) " << std::round(etaDays * 10.0) / 10.0 << " days\n" << std::flush;
    }

    for (std::thread& t : workers) t.join();
    const MetaCounters finalMeta{resumedMeta.games + g_totalGames.load(),
                                 resumedMeta.tbAdjudications + g_totalTbAdjudications.load()};
    saveMeta(metaPath, finalMeta);
    std::cout << "[datagen] stopped. total positions "
              << fmtCount(resumedPositions + g_totalPositions.load())
              << " (+" << fmtCount(g_totalPositions.load()) << " this run)"
              << "  games " << fmtCount(finalMeta.games)
              << "  tb-adj " << fmtCount(finalMeta.tbAdjudications) << "\n";
    return 0;
}

int runDatagenDump(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: ./chess datagen-dump <file.bin> [count]\n";
        return 1;
    }
    const char* path = argv[2];
    const uint64_t count = (argc >= 4) ? std::strtoull(argv[3], nullptr, 10) : 10;

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "datagen-dump: cannot open " << path << "\n";
        return 1;
    }
    const auto bytes = static_cast<uint64_t>(in.tellg());
    in.seekg(0);
    std::cout << path << ": " << bytes / sizeof(BulletRecord) << " records ("
              << bytes << " bytes";
    if (bytes % sizeof(BulletRecord) != 0) {
        std::cout << ", WARNING: not a multiple of 32";
    }
    std::cout << ")\n";

    // FEN is printed in the stm frame: the side to move is always White here,
    // so score/result read as "for the printed White".
    BulletRecord r;
    for (uint64_t i = 0; i < count
         && in.read(reinterpret_cast<char*>(&r), sizeof(r)); ++i) {
        std::cout << i << ": " << recordToFen(r)
                  << " | score " << r.score
                  << " | result " << (static_cast<double>(r.result) / 2.0)
                  << " | ksq " << static_cast<int>(r.ksq)
                  << " oppKsq " << static_cast<int>(r.oppKsq) << "\n";
    }
    return 0;
}

} // namespace NNUE
