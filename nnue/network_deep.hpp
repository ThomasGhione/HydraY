#pragma once

// Rete quantizzata con layer intermedi (progetto "layer intermedio").
//
//   (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 1
//
// UN SOLO layer intermedio: vedi trainer_deep.rs per il perche' si e' scelto di
// non replicare il 16->32->1 dell'esempio di bullet.
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
//   l2w [8][1][16]     f32
//   l2b [8]            f32
//   payload 6.425.632 B, file 6.425.664 B
//
// ARITMETICA (deve combaciare con sanity_deep.rs riga per riga):
//
//   c[i] = clamp(acc[i], 0, QA)                  scala QA, sta in u8
//   p[j] = c[j] * c[j+512] / QA                  divisione INTERA troncata
//   z[o] = SUM(p[i] * l1w[o][i]) / (QA*QB) + l1b[o]   divisione in FLOAT
//   a[o] = clamp(z[o], 0, 1)^2                   SCReLU in float
//   y    = SUM a[o]*l2w[o] + l2b                 lineare
//   cp   = round(y * SCALE)
//
// La divisione di p e' intera e quella di z e' in virgola mobile: non e' una
// svista, e' quello che fa l'oracolo. Uniformarle cambierebbe i risultati.

#include <cstddef>
#include <cstdint>

#include "network.hpp"

namespace NNUE::Deep {

inline constexpr int INPUTS        = 768;
inline constexpr int HIDDEN        = 1024;
inline constexpr int L1_SIZE       = 16;
// Deve coincidere con NNUE::INPUT_BUCKETS: l'accumulatore reinterpreta il
// prefisso l0 della rete profonda come una Network (vedi nnue.cpp).
inline constexpr int INPUT_BUCKETS = NNUE::INPUT_BUCKETS;
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
    float   l2w[OUTPUT_BUCKETS][L1_SIZE];
    float   l2b[OUTPUT_BUCKETS];

    // Copia di l1w gia' allargata a i16, riempita al caricamento. Serve solo al
    // percorso SENZA AVX-VNNI, dove il prodotto scalare lavora a corsie i16:
    // convertire i pesi a ogni valutazione costava una cvtepi8_epi16 per
    // istruzione utile. 256 KiB in piu' in cambio di un ciclo interno pulito.
    // NON sta nel file: e' derivata, e va rigenerata a ogni caricamento.
    alignas(64) int16_t l1w16[OUTPUT_BUCKETS][L1_SIZE][HIDDEN];

    // Pesi TRASPOSTI per il percorso sparso (solo AVX-VNNI). L'85% dell'ingresso
    // di l1 e' zero, ma il ciclo guidato dalle uscite moltiplica per zero lo
    // stesso. Guidandolo dagli ingressi NON nulli servono, per un dato gruppo di
    // quattro ingressi, i pesi verso tutte e 16 le uscite: 16*4 = 64 byte
    // contigui. I primi 32 sono le uscite 0-7 (quattro pesi ciascuna), i secondi
    // 32 le uscite 8-15, cioe' esattamente i due operandi di vpdpbusd.
    //
    // L'indice di riga e' il gruppo da quattro nell'ordine in cui il pairwise
    // SCRIVE, che non e' quello di l1w: packus lavora per corsia da 128 bit e
    // scambia le due meta' centrali di ogni blocco da 32. Assorbire quello
    // scambio qui e' gratis (la tabella si costruisce comunque) e toglie una
    // vpermq ogni 32 uscite dal ciclo caldo.
    //
    // Derivata, non letta dal file. 128 KiB.
    alignas(64) int8_t l1wT[OUTPUT_BUCKETS][HIDDEN / 4][L1_SIZE * 4];
};

// Ordine in cui packus_epi16 lascia i 32 byte di un blocco rispetto alla
// sorgente: le due meta' centrali risultano scambiate. Serve a l1wT e al
// controllo di coerenza fra i due percorsi.
inline constexpr int PACKUS_MAP[32] = {
     0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
};

// Taglia del payload sul file, indipendente dal padding della struct: e' il
// numero che il loader deve leggere, e va calcolato dai campi e non da
// sizeof(NetworkDeep), che l'alignas puo' gonfiare.
inline constexpr size_t PAYLOAD_BYTES =
      static_cast<size_t>(INPUT_BUCKETS) * INPUTS * HIDDEN * sizeof(int16_t)
    + static_cast<size_t>(HIDDEN) * sizeof(int16_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * HIDDEN * sizeof(int8_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * sizeof(float);
// 12'717'088 a 8 bucket d'ingresso (era 6'425'632 a 4): il valore segue
// INPUT_BUCKETS. Nota: le reti profonde gia' addestrate sono a 4 bucket e
// su questo branch non sono caricabili -- qui si misura solo il layer singolo.
static_assert(PAYLOAD_BYTES == 12'717'088, "layout cambiato: aggiorna sanity_deep.rs");

// Forward scalare, dagli accumulatori delle due prospettive alla valutazione
// in centipedine. Riferimento di correttezza per la versione vettoriale.
[[nodiscard]] int32_t forwardScalar(const NetworkDeep& net,
                                    const int16_t* accStm,
                                    const int16_t* accNtm,
                                    int outputBucket) noexcept;

// Stessa funzione, vettoriale. DEVE restituire lo stesso valore di
// forwardScalar per ogni ingresso: e' un'ottimizzazione, non un'altra rete.
[[nodiscard]] int32_t forwardSimd(const NetworkDeep& net,
                                  const int16_t* accStm,
                                  const int16_t* accNtm,
                                  int outputBucket) noexcept;

// true se il binario e' stato compilato con AVX-VNNI (vpdpbusd), che raddoppia
// le moltiplicazioni per istruzione nel layer l1. Senza, forwardSimd usa un
// percorso a corsie i16 che e' esatto ma costa il doppio delle istruzioni.
[[nodiscard]] bool hasVnniPath() noexcept;

// Carica un file quantised.bin nel formato sopra. Ritorna false (senza
// scrivere in `net`) se taglia o padding non tornano.
[[nodiscard]] bool loadFromFile(const char* path, NetworkDeep& net) noexcept;

} // namespace NNUE::Deep
