#include "datagen.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
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

// Defaults and quality filters. A position whose best move captures or
// promotes is skipped: its score hinges on unresolved tactics.
constexpr int      DEFAULT_THREADS          = 3;
constexpr uint64_t DEFAULT_NODES_PER_MOVE   = 8000;
// 50M positions = 1.6 GB, about a day of a 12-thread machine.
constexpr uint64_t DEFAULT_SHARD_POSITIONS  = 50'000'000;
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
// Only the ETA line reads this; --target sets it for a short batch.
constexpr uint64_t TARGET_POSITIONS_DEFAULT = 5'000'000'000;
uint64_t           g_targetPositions        = TARGET_POSITIONS_DEFAULT;
// WDL ignores the fifty-move counter: a decisive result counts only below this clock.
constexpr int      DECISIVE_TB_ADJ_MAX_HMC  = 60;

// Endgame seeding fills output bucket 0 (2-5 men), which self-play never
// reaches: Syzygy adjudication ends the game first, a won ending scores past
// MAX_RECORD_SCORE_CP, and seeded endings are decided before MIN_RECORD_PLY.
// Seeded games record from ply 0, clamp their scores and skip adjudication.
// One game in --eg-every is seeded; 1 makes a dedicated batch,
// which fills the bucket far faster than a mixed run.
constexpr int      ENDGAME_SEED_EVERY_DEFAULT = 8;
int                g_endgameSeedEvery = ENDGAME_SEED_EVERY_DEFAULT;
constexpr int      ENDGAME_SEED_MIN_MEN = 3; // K+K plus one man
constexpr int      ENDGAME_SEED_MAX_MEN = 6; // kept inside bucket 0 (2-5 men) plus one
constexpr int      ENDGAME_SEED_TRIES = 64;
// Most seeds give every extra man to one side: with independent colours the
// material is balanced, and balanced endings are almost all draws.
constexpr int      ENDGAME_SEED_LOPSIDED_PCT = 65;
// A drawn ending runs to the fifty-move rule while a decided one ends after
// WIN_ADJ_PLIES: the cap keeps the draws from drowning the rest.
constexpr size_t   ENDGAME_SEED_MAX_RECORDS = 8;

std::atomic<bool>     g_stop{false};
std::atomic<uint64_t> g_totalPositions{0};
std::atomic<uint64_t> g_totalGames{0};
std::atomic<uint64_t> g_totalTbAdjudications{0};
std::atomic<uint64_t> g_totalEndgameSeeded{0};
// Games started but never written: an opening past OPENING_MAX_IMBALANCE_CP,
// or an endgame seed that found no legal position.
std::atomic<uint64_t> g_discardedOpenings{0};
std::atomic<uint64_t> g_discardedSeeds{0};

// Benchmark mode (--seed, --games): each game starts
// from its own RNG stream, an empty TT and empty histories, so a run replays
// identically at any thread count.
uint64_t              g_benchSeed = 0;   // 0 = off
uint64_t              g_gameLimit = 0;   // 0 = unlimited
std::atomic<uint64_t> g_nextGameId{0};
std::atomic<int>      g_activeWorkers{0};

// splitmix64: consecutive game ids must give uncorrelated streams.
uint64_t mixSeed(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Shared and read-only after load: WDL probes are thread-safe.
syzygy::SyzygyProber g_syzygy;

void onStopSignal(int) { g_stop.store(true, std::memory_order_release); }

// All workers append whole games to one shard, <out>.sNNNN.bin.part. At
// shardSize positions it is sealed (renamed to .bin) and the next one opened,
// so a .bin file never grows again: only sealed shards get uploaded or
// trained on. A restart appends to the open .part. shardSize 0 = one
// unsealed <out>.bin, for short experiments.
class ShardWriter {
public:
    bool open(const std::string& prefix, uint64_t shardSize) {
        namespace fs = std::filesystem;
        prefix_ = prefix;
        shardSize_ = shardSize;
        if (shardSize_ == 0) return openFile(prefix_ + ".bin");

        const fs::path p(prefix_);
        const fs::path dir = p.parent_path().empty() ? fs::path(".") : p.parent_path();
        const std::string stem = p.filename().string() + ".s";
        int nextIndex = 0;
        int partIndex = -1;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            const std::string name = entry.path().filename().string();
            if (!name.starts_with(stem)) continue;
            const std::string_view rest = std::string_view(name).substr(stem.size());
            const bool part = rest.ends_with(".bin.part");
            if (!part && !rest.ends_with(".bin")) continue;
            const std::string_view digits = rest.substr(0, rest.find('.'));
            if (digits.empty() || digits.find_first_not_of("0123456789") != std::string_view::npos) continue;
            const int index = std::atoi(std::string(digits).c_str());
            nextIndex = std::max(nextIndex, index + 1);
            if (part) partIndex = std::max(partIndex, index);
        }
        index_ = (partIndex >= 0) ? partIndex : nextIndex;
        if (!openFile(partPath())) return false;
        inShard_ = static_cast<uint64_t>(out_.tellp()) / sizeof(BulletRecord);
        return inShard_ < shardSize_ || rotate();
    }

    // One game per call, so a game never straddles two shards.
    bool write(const std::vector<BulletRecord>& records) {
        std::lock_guard lock(mutex_);
        out_.write(reinterpret_cast<const char*>(records.data()),
                   static_cast<std::streamsize>(records.size() * sizeof(BulletRecord)));
        out_.flush();
        if (!out_) return false;
        inShard_ += records.size();
        return shardSize_ == 0 || inShard_ < shardSize_ || rotate();
    }

    // At exit: an empty .part is removed, a partial one sealed when asked.
    void close(bool seal) {
        std::lock_guard lock(mutex_);
        out_.close();
        if (shardSize_ == 0) return;
        std::error_code ec;
        if (inShard_ == 0) std::filesystem::remove(partPath(), ec);
        else if (seal) std::filesystem::rename(partPath(), sealedPath(), ec);
    }

    std::string pattern() const {
        return shardSize_ == 0 ? prefix_ + ".bin" : prefix_ + ".sNNNN.bin";
    }
    std::string currentPath() const {
        return shardSize_ == 0 ? prefix_ + ".bin" : partPath();
    }

private:
    bool openFile(const std::string& path) {
        out_.open(path, std::ios::binary | std::ios::app);
        if (!out_) std::cerr << "datagen: cannot open " << path << " for append.\n";
        return static_cast<bool>(out_);
    }

    bool rotate() {
        out_.close();
        std::error_code ec;
        std::filesystem::rename(partPath(), sealedPath(), ec);
        if (ec) {
            std::cerr << "datagen: cannot seal " << partPath() << ": " << ec.message() << "\n";
            return false;
        }
        std::cout << "[datagen] sealed " << sealedPath() << "\n" << std::flush;
        ++index_;
        inShard_ = 0;
        return openFile(partPath());
    }

    std::string sealedPath() const {
        char index[16];
        std::snprintf(index, sizeof(index), "%04d", index_);
        return prefix_ + ".s" + index + ".bin";
    }
    std::string partPath() const { return sealedPath() + ".part"; }

    std::mutex mutex_;
    std::ofstream out_;
    std::string prefix_;
    uint64_t shardSize_ = 0;
    int index_ = 0;
    uint64_t inShard_ = 0;
};

ShardWriter g_writer;

// Everything one worker owns.
struct WorkerContext {
    engine::SearchRuntime runtime;
    TT tt;
    std::atomic<bool> interrupted{false};
    std::mt19937_64 rng;

    explicit WorkerContext(uint64_t seed) : rng(seed) {
        runtime.transpositionTable = &tt;
        // The node cap reports a truncated iteration through this flag.
        runtime.searchInterrupted = &interrupted;
        runtime.stopSearchRequested = &g_stop;
    }
};

// A random legal position of ENDGAME_SEED_MIN_MEN..MAX_MEN men, or "" when
// every try came out illegal or a dead draw. Squares run 0 = a8 .. 63 = h1.
std::string randomEndgameFen(std::mt19937_64& rng) {
    using chess::Board;
    // Weighted towards the pieces that actually decide endgames, while still
    // producing the heavy-piece cases (KQvK) that expose the empty bucket.
    static constexpr char POOL[] = "PPPPPRRRNNBBQ";
    static constexpr int  POOL_SIZE = static_cast<int>(sizeof(POOL)) - 1;

    for (int attempt = 0; attempt < ENDGAME_SEED_TRIES; ++attempt) {
        std::array<char, 64> sq;
        sq.fill('.');

        const int wk = static_cast<int>(rng() % 64);
        const int bk = static_cast<int>(rng() % 64);
        const int fileGap = std::abs((wk % 8) - (bk % 8));
        const int rankGap = std::abs((wk / 8) - (bk / 8));
        if (fileGap <= 1 && rankGap <= 1) continue; // kings touching (or equal)
        sq[static_cast<size_t>(wk)] = 'K';
        sq[static_cast<size_t>(bk)] = 'k';

        const int men = ENDGAME_SEED_MIN_MEN
            + static_cast<int>(rng() % (ENDGAME_SEED_MAX_MEN - ENDGAME_SEED_MIN_MEN + 1));
        const bool lopsided = (rng() % 100) < ENDGAME_SEED_LOPSIDED_PCT;
        const bool strongIsWhite = (rng() & 1) != 0;
        for (int placed = 0; placed < men - 2; ++placed) {
            const char piece = POOL[rng() % static_cast<uint64_t>(POOL_SIZE)];
            const bool white = lopsided ? strongIsWhite : ((rng() & 1) != 0);
            // Pawns cannot stand on the first or last rank.
            const int lo = (piece == 'P') ? 8 : 0;
            const int hi = (piece == 'P') ? 56 : 64;
            for (int probe = 0; probe < 32; ++probe) {
                const int s = lo + static_cast<int>(rng() % static_cast<uint64_t>(hi - lo));
                if (sq[static_cast<size_t>(s)] != '.') continue;
                sq[static_cast<size_t>(s)] = white
                    ? piece : static_cast<char>(piece - 'A' + 'a');
                break;
            }
        }

        // No castling rights and no en passant: both would be unsound on a
        // position assembled at random, and TB probes assume the former.
        const std::string fen = chess::placementFen(sq) + (((rng() & 1) != 0) ? " w - - 0 1" : " b - - 0 1");

        const Board b{fen};
        // A position where the side that just moved is left in check is illegal.
        if (b.inCheck(Board::oppositeColor(b.getActiveColor()))) continue;
        // Dead draws (K vs K, K vs KN/KB) end before recording anything.
        if (b.hasInsufficientMaterialDraw()) continue;
        return fen;
    }
    return {};
}

// Set by --start chess324. A run holds one start regime, so pools
// keep separate prefixes and are mixed at shuffle time.
bool g_chess324 = false;

// The DFRC subset with king and rooks on their usual squares, so standard
// castling still applies. Each side is drawn independently.
std::string randomChess324Fen(std::mt19937_64& rng) {
    const auto backRank = [&rng](bool white) {
        std::string rank = "r...k..r";
        // One bishop on c/g and one on b/d/f: opposite colours.
        rank[(rng() & 1) ? 2u : 6u] = 'b';
        rank[1 + 2 * (rng() % 3)] = 'b';
        const uint64_t queen = rng() % 3;
        uint64_t slot = 0;
        for (char& c : rank) {
            if (c == '.') c = (slot++ == queen) ? 'q' : 'n';
            if (white) c = static_cast<char>(c - 'a' + 'A');
        }
        return rank;
    };
    return backRank(false) + "/pppppppp/8/8/8/8/PPPPPPPP/" + backRank(true) + " w KQkq - 0 1";
}

// Plays one self-play game; returns the number of positions written
// (0 = game discarded: unbalanced opening, dead-end opening, or stop request).
uint64_t playOneGame(WorkerContext& w, uint64_t nodesPerMove, bool endgameSeed) {
    using chess::Board;

    std::string seedFen;
    if (endgameSeed) {
        seedFen = randomEndgameFen(w.rng);
        if (seedFen.empty()) {
            g_discardedSeeds.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    } else if (g_chess324) {
        seedFen = randomChess324Fen(w.rng);
    }
    Board b = seedFen.empty() ? Board{} : Board{seedFen};

    // A seeded game has no random opening walk to skip.
    const int openingPlies = endgameSeed
        ? 0 : OPENING_PLIES_MIN + static_cast<int>(w.rng() & 1);
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

        // Syzygy adjudication: a game in TB range ends with the exact result
        // (cursed wins and blessed losses are draws). Not for seeded games,
        // which would end before recording anything.
        if (!endgameSeed && b.castlingRights() == 0) {
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
            }
        }

        w.tt.incrementGeneration();
        w.runtime.nodesSearched = 0;
        w.runtime.maxNodes = nodesPerMove;
        w.runtime.clearInterrupted();
        const auto res =
            engine::Searcher::runIterativeDeepening(b, w.runtime, 1, TARGET_DEPTH_CAP);

        // Depth 1 unfinished within the node budget: play the first move, record nothing.
        const bool searchOk = res.completedAnyDepth && !(res.bestMove == chess::Move{});
        const chess::Move best = searchOk ? res.bestMove : legal[0];
        const int32_t stmScore = res.bestScore;
        const int32_t whiteScore = (active == Board::WHITE) ? stmScore : -stmScore;

        // Seeded endgames are lopsided by construction.
        if (!endgameSeed && ply == openingPlies
            && std::abs(whiteScore) > OPENING_MAX_IMBALANCE_CP) {
            g_discardedOpenings.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        const engine::MoveTraits traits =
            engine::moveTraits(b, best, b.getEnPassant(), Board::promotionRank(active == Board::WHITE));
        const bool isTactical = traits.isCapture || traits.isPromotion;
        // Seeded games clamp a mate score instead of dropping it: "completely
        // winning" is the right target for a won ending.
        const bool inScoreRange = std::abs(whiteScore) <= MAX_RECORD_SCORE_CP;
        const bool seedQuotaLeft =
            !endgameSeed || pending.size() < ENDGAME_SEED_MAX_RECORDS;
        if (searchOk && ply >= (endgameSeed ? 0 : MIN_RECORD_PLY) && !isTactical
            && !b.inCheck(active) && (inScoreRange || endgameSeed) && seedQuotaLeft) {
            const int32_t target =
                std::clamp(whiteScore, -MAX_RECORD_SCORE_CP, MAX_RECORD_SCORE_CP);
            pending.push_back(packPosition(b, static_cast<int16_t>(target)));
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
        p.record.result = static_cast<uint8_t>(p.blackToMove ? 2 - whiteResultX2 : whiteResultX2);
        flat.push_back(p.record);
    }
    if (!flat.empty()) {
        if (!g_writer.write(flat)) {
            std::cerr << "datagen: write failed (disk full?) - stopping.\n";
            g_stop.store(true, std::memory_order_release);
            return 0;
        }
    }

    g_totalPositions.fetch_add(flat.size(), std::memory_order_relaxed);
    g_totalGames.fetch_add(1, std::memory_order_relaxed);
    if (endgameSeed) g_totalEndgameSeeded.fetch_add(1, std::memory_order_relaxed);
    return flat.size();
}

void workerLoop(int threadId, uint64_t nodesPerMove) {
    const uint64_t seed = std::random_device{}()
        ^ (static_cast<uint64_t>(
               std::chrono::steady_clock::now().time_since_epoch().count())
           << 16)
        ^ (static_cast<uint64_t>(threadId) << 56);
    WorkerContext w(seed);
    while (!g_stop.load(std::memory_order_acquire)) {
        // One global counter: game N is the same game at any thread count.
        const uint64_t gameId = g_nextGameId.fetch_add(1, std::memory_order_relaxed);
        if (g_gameLimit != 0 && gameId >= g_gameLimit) break;
        if (g_benchSeed != 0) {
            w.rng.seed(mixSeed(g_benchSeed + gameId));
            w.tt.clear();
            w.runtime.clearHistory();
        }
        playOneGame(w, nodesPerMove,
                    gameId % static_cast<uint64_t>(g_endgameSeedEvery) == 0);
    }
    g_activeWorkers.fetch_sub(1, std::memory_order_release);
}

// Positions already on disk for this prefix: shards, the open .part and the
// <out>.t<N>.bin files of older runs. The counters resume from the true total
// instead of restarting at 0.
uint64_t countExistingPositions(const std::string& outPrefix) {
    namespace fs = std::filesystem;
    const fs::path p(outPrefix);
    const fs::path dir = p.parent_path().empty() ? fs::path(".") : p.parent_path();
    const std::string stem = p.filename().string() + ".";
    uint64_t bytes = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with(stem) && (name.ends_with(".bin") || name.ends_with(".bin.part"))) {
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

constexpr const char* DATAGEN_USAGE =
    "usage: ./chess datagen [options]\n"
    "  --out PREFIX     output prefix (default nnue/data/hydray)\n"
    "  --threads N      worker threads, 1..32 (default 3)\n"
    "  --nodes N        nodes per move, at least 256 (default 8000)\n"
    "  --net FILE       labeler network (default: the embedded net)\n"
    "  --tb DIR         Syzygy tablebases (default engine/syzygy/files)\n"
    "  --shard N        positions per shard, 0 = one unsealed PREFIX.bin\n"
    "                   (default 50000000)\n"
    "  --seal-on-exit   seal the partial shard when the run stops\n"
    "  --start MODE     standard or chess324 (default standard)\n"
    "  --eg-every N     seed 1 game in N from a random ending, 1..1000 (default 8)\n"
    "  --games N        stop after N games started, discarded openings included\n"
    "                   (default: until Ctrl+C / SIGTERM)\n"
    "  --target N       positions for the ETA line (default 5000000000)\n"
    "  --seed N         benchmark mode: game N replays at any thread count\n";

// Whole decimal strings only: "8k" or "-1" must not become 8 or 2^64-1.
bool parseCount(const char* text, uint64_t& value) {
    if (text == nullptr || *text < '0' || *text > '9') return false;
    char* end = nullptr;
    errno = 0;
    value = std::strtoull(text, &end, 10);
    return errno == 0 && *end == '\0';
}

struct DatagenOptions {
    std::string outPrefix = "nnue/data/hydray";
    int threads = DEFAULT_THREADS;
    uint64_t nodesPerMove = DEFAULT_NODES_PER_MOVE;
    const char* netPath = nullptr;
    std::string tbPath = "engine/syzygy/files";
    uint64_t shardSize = DEFAULT_SHARD_POSITIONS;
    bool sealOnExit = false;
    bool help = false;
};

// Sets the options and the g_* knobs; false after printing why it failed.
bool parseDatagenArgs(int argc, char* argv[], DatagenOptions& o) {
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        const char* value = nullptr;
        if (const size_t eq = flag.find('='); flag.starts_with("--") && eq != std::string::npos) {
            value = argv[i] + eq + 1;
            flag.resize(eq);
        }
        if (flag == "--help" || flag == "-h") {
            o.help = true;
            return true;
        }
        if (flag == "--seal-on-exit") {
            o.sealOnExit = true;
            continue;
        }
        if (!flag.starts_with("--")) {
            std::cerr << "datagen: unexpected argument '" << flag
                      << "' (the positional form became flags)\n" << DATAGEN_USAGE;
            return false;
        }
        if (value == nullptr) {
            if (i + 1 >= argc) {
                std::cerr << "datagen: " << flag << " needs a value\n";
                return false;
            }
            value = argv[++i];
        }

        uint64_t n = 0;
        const auto count = [&](uint64_t lo, uint64_t hi) {
            if (parseCount(value, n) && n >= lo && n <= hi) return true;
            std::cerr << "datagen: " << flag << " must be a whole number in "
                      << lo << ".." << hi << ", got '" << value << "'\n";
            return false;
        };
        constexpr uint64_t ANY = UINT64_MAX;
        if (flag == "--out") {
            o.outPrefix = value;
        } else if (flag == "--threads") {
            if (!count(1, 32)) return false;
            o.threads = static_cast<int>(n);
        } else if (flag == "--nodes") {
            if (!count(256, ANY)) return false;
            o.nodesPerMove = n;
        } else if (flag == "--net") {
            o.netPath = value;
        } else if (flag == "--tb") {
            o.tbPath = value;
        } else if (flag == "--shard") {
            if (!count(0, ANY)) return false;
            o.shardSize = n;
        } else if (flag == "--start") {
            // Rejected, not ignored: a typo would fill a Chess324 pool with standard games.
            const std::string mode = value;
            if (mode != "standard" && mode != "chess324") {
                std::cerr << "datagen: --start must be standard or chess324\n";
                return false;
            }
            g_chess324 = (mode == "chess324");
        } else if (flag == "--eg-every") {
            if (!count(1, 1000)) return false;
            g_endgameSeedEvery = static_cast<int>(n);
        } else if (flag == "--games") {
            if (!count(1, ANY)) return false;
            g_gameLimit = n;
        } else if (flag == "--target") {
            if (!count(1, ANY)) return false;
            g_targetPositions = n;
        } else if (flag == "--seed") {
            if (!count(1, ANY)) return false;
            g_benchSeed = n;
        } else {
            std::cerr << "datagen: unknown option " << flag << "\n" << DATAGEN_USAGE;
            return false;
        }
    }
    return true;
}

} // namespace

int runDatagen(int argc, char* argv[]) {
    DatagenOptions opts;
    if (!parseDatagenArgs(argc, argv, opts)) return 1;
    if (opts.help) {
        std::cout << DATAGEN_USAGE;
        return 0;
    }
    const std::string& outPrefix = opts.outPrefix;
    const int threads = opts.threads;
    const uint64_t nodesPerMove = opts.nodesPerMove;

    // Labeler network: the given path, else the embedded net. A bad path fails
    // hard: falling back to another net would mislabel days of data.
    const char* netPath = opts.netPath;
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

    // Without tables the endgames are simply played out.
    const std::string& tbPath = opts.tbPath;
    const bool tbOn = g_syzygy.load(tbPath) && g_syzygy.maxPieces() >= 3;

    const uint64_t resumedPositions = countExistingPositions(outPrefix);
    const std::string metaPath = outPrefix + ".meta";
    const MetaCounters resumedMeta = loadMeta(metaPath);
    if (!g_writer.open(outPrefix, opts.shardSize)) return 1;

    std::cout << "HydraY datagen - bulletformat self-play data\n"
              << "  output : " << g_writer.pattern()
              << (opts.shardSize != 0
                     ? ", sealed every " + fmtCount(opts.shardSize) + " positions"
                       + (opts.sealOnExit ? " and at exit" : "")
                     : std::string(" (never sealed)"))
              << "\n           writing " << g_writer.currentPath() << " (append)\n"
              << "  resume : " << fmtCount(resumedPositions)
              << " positions already on disk for this prefix\n"
              << "  labels : NNUE (" << (netPath != nullptr ? netPath : "embedded")
              << ", eval scale x" << evalScale() << ")\n"
              << "  threads: " << threads << "  nodes/move: " << nodesPerMove << "\n"
              << "  filters: ply>=" << MIN_RECORD_PLY << ", not in check, quiet bestmove, |cp|<="
              << MAX_RECORD_SCORE_CP << "\n"
              << "  target : " << fmtCount(g_targetPositions)
              << " positions (ETA line only; --target)\n"
              << "  start  : " << (g_chess324 ? "random Chess324 setup" : "standard position")
              << " + " << OPENING_PLIES_MIN << "-" << OPENING_PLIES_MIN + 1
              << " random plies (--start)\n"
              << "  endgame: " << (g_endgameSeedEvery == 1
                     ? std::string("EVERY game")
                     : std::string("1 game in ") + std::to_string(g_endgameSeedEvery))
              << " seeded from a random "
              << ENDGAME_SEED_MIN_MEN << "-" << ENDGAME_SEED_MAX_MEN
              << "-man position (fills output bucket 0)"
              << (g_endgameSeedEvery == 1 ? "  [DEDICATED BATCH]" : "") << "\n"
              << "  syzygy : " << (tbOn
                     ? std::string("adjudication ON (") + tbPath + ", "
                       + std::to_string(g_syzygy.maxPieces()) + "-man)"
                     : "OFF (no tablebases found - endgames played out)") << "\n"
              << "  stop   : Ctrl+C / SIGTERM (partial games are discarded)\n"
              << std::flush;

    if (g_benchSeed != 0 || g_gameLimit != 0) {
        std::cout << "  bench  : seed " << g_benchSeed
                  << (g_benchSeed != 0 ? " (TT and history reset per game)" : " (off)")
                  << ", games "
                  << (g_gameLimit != 0 ? std::to_string(g_gameLimit) : std::string("unlimited"))
                  << "\n" << std::flush;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    g_activeWorkers.store(threads, std::memory_order_release);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back(workerLoop, t, nodesPerMove);
    }

    const auto start = std::chrono::steady_clock::now();
    auto lastReport = start;
    while (!g_stop.load(std::memory_order_acquire)
           && g_activeWorkers.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport < std::chrono::seconds(30)) continue;
        lastReport = now;

        const uint64_t generated = g_totalPositions.load(std::memory_order_relaxed);
        const uint64_t total = resumedPositions + generated;
        const double elapsed = std::chrono::duration<double>(now - start).count();
        // The rate is this run's; the total includes what was already on disk.
        const double rate = (elapsed > 0.0) ? static_cast<double>(generated) / elapsed : 0.0;
        const double etaDays = (rate > 0.0)
            ? static_cast<double>(g_targetPositions - std::min(total, g_targetPositions))
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
                  << "  eg-seed "
                  << fmtCount(g_totalEndgameSeeded.load(std::memory_order_relaxed))
                  << "  discarded " << fmtCount(g_discardedOpenings.load(std::memory_order_relaxed)
                                                + g_discardedSeeds.load(std::memory_order_relaxed))
                  << "  pos/s " << static_cast<uint64_t>(rate)
                  << "  ETA(" << fmtCount(g_targetPositions) << ") "
                  << std::round(etaDays * 10.0) / 10.0 << " days\n" << std::flush;
    }

    for (std::thread& t : workers) t.join();
    g_writer.close(opts.sealOnExit);
    const MetaCounters finalMeta{resumedMeta.games + g_totalGames.load(),
                                 resumedMeta.tbAdjudications + g_totalTbAdjudications.load()};
    saveMeta(metaPath, finalMeta);
    std::cout << "[datagen] stopped. total positions "
              << fmtCount(resumedPositions + g_totalPositions.load())
              << " (+" << fmtCount(g_totalPositions.load()) << " this run)"
              << "  games " << fmtCount(finalMeta.games)
              << "  tb-adj " << fmtCount(finalMeta.tbAdjudications)
              << "  discarded this run: " << g_discardedOpenings.load() << " openings past "
              << OPENING_MAX_IMBALANCE_CP << " cp, " << g_discardedSeeds.load()
              << " endgame seeds without a legal position\n";
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
