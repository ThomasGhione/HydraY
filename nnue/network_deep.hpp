#pragma once

// Rete quantizzata con layer intermedi (progetto "layer intermedio").
//
//   (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 32 -> 1
//
// Gli INPUT sono identici alla rete a un layer (network.hpp): stessi 4 king
// bucket specchiati, stessa formula della feature, stesso accumulatore i16.
// Cambia solo cio' che succede DOPO l'accumulatore, quindi accumulator.hpp non
// va toccato.
//
// Questo header e' deliberatamente SEPARATO da network.hpp: la rete a un layer
// resta quella in produzione finche' questa non vince un SPRT, e le due devono
// poter coesistere senza #ifdef sparsi nel motore.
//
// Il lettore di riferimento e' nnue/trainer/src/bin/sanity_deep.rs. Quando i
// due divergono, ha ragione sanity_deep: e' l'oracolo, e' piu' semplice, ed e'
// scritto contro il formato di bullet.
//
// LAYOUT DEL FILE (little-endian, pad a 64 B con "bullet" ripetuto):
//
//   l0w [4*768][1024]  i16  QA=255   (factoriser gia' sommato al salvataggio)
//   l0b [1024]         i16  QA
//   l1w [8][16][1024]  i8   QB=64    (trasposto: ogni bucket contiguo)
//   l1b [8][16]        f32  scala reale
//   l2w [8][32][16]    f32
//   l2b [8][32]        f32
//   l3w [8][1][32]     f32
//   l3b [8]            f32
//   payload 6.443.552 B, file 6.443.584 B
//
// ARITMETICA (deve combaciare con sanity_deep.rs riga per riga):
//
//   c[i] = clamp(acc[i], 0, QA)                  scala QA, sta in u8
//   p[j] = c[j] * c[j+512] / QA                  divisione INTERA troncata
//   z[o] = SUM(p[i] * l1w[o][i]) / (QA*QB) + l1b[o]   divisione in FLOAT
//   a[o] = clamp(z[o], 0, 1)^2                   SCReLU in float
//   ... l2, l3 ...
//   cp   = round(y * SCALE)
//
// La divisione di p e' intera e quella di z e' in virgola mobile: non e' una
// svista, e' quello che fa l'oracolo. Uniformarle cambierebbe i risultati.

#include <cstddef>
#include <cstdint>

namespace NNUE::Deep {

inline constexpr int INPUTS        = 768;
inline constexpr int HIDDEN        = 1024;
inline constexpr int L1_SIZE       = 16;
inline constexpr int L2_SIZE       = 32;
inline constexpr int INPUT_BUCKETS = 4;
inline constexpr int OUTPUT_BUCKETS = 8;
inline constexpr int32_t QA    = 255;
inline constexpr int32_t QB    = 64;
inline constexpr int32_t SCALE = 400;

// Uscita del pairwise per prospettiva; l'ingresso di l1 e' HIDDEN (due
// prospettive concatenate), non 2*HIDDEN — e' il motivo per cui il layer costa
// poco.
inline constexpr int PAIRWISE_OUT = HIDDEN / 2;
static_assert(2 * PAIRWISE_OUT == HIDDEN, "il pairwise richiede HIDDEN pari");

struct alignas(64) NetworkDeep {
    int16_t l0w[INPUT_BUCKETS * INPUTS][HIDDEN];
    int16_t l0b[HIDDEN];
    int8_t  l1w[OUTPUT_BUCKETS][L1_SIZE][HIDDEN];
    float   l1b[OUTPUT_BUCKETS][L1_SIZE];
    float   l2w[OUTPUT_BUCKETS][L2_SIZE][L1_SIZE];
    float   l2b[OUTPUT_BUCKETS][L2_SIZE];
    float   l3w[OUTPUT_BUCKETS][L2_SIZE];
    float   l3b[OUTPUT_BUCKETS];
};

// Taglia del payload sul file, indipendente dal padding della struct: e' il
// numero che il loader deve leggere, e va calcolato dai campi e non da
// sizeof(NetworkDeep), che l'alignas puo' gonfiare.
inline constexpr size_t PAYLOAD_BYTES =
      static_cast<size_t>(INPUT_BUCKETS) * INPUTS * HIDDEN * sizeof(int16_t)
    + static_cast<size_t>(HIDDEN) * sizeof(int16_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * HIDDEN * sizeof(int8_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L2_SIZE * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L2_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L2_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * sizeof(float);
static_assert(PAYLOAD_BYTES == 6'443'552, "layout cambiato: aggiorna sanity_deep.rs");

// Forward scalare, dagli accumulatori delle due prospettive alla valutazione
// in centipedine. Riferimento di correttezza per la versione vettoriale.
[[nodiscard]] int32_t forwardScalar(const NetworkDeep& net,
                                    const int16_t* accStm,
                                    const int16_t* accNtm,
                                    int outputBucket) noexcept;

// Carica un file quantised.bin nel formato sopra. Ritorna false (senza
// scrivere in `net`) se taglia o padding non tornano.
[[nodiscard]] bool loadFromFile(const char* path, NetworkDeep& net) noexcept;

} // namespace NNUE::Deep
