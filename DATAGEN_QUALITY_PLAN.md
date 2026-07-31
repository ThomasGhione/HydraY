# DATAGEN_QUALITY_PLAN.md — dedup offline del dataset

Data: 2026-07-13. Contesto: datagen v3 @ 8k in corso (rete ob). Cross-reference:
`NNUE_PLAN.md` (§ Miglioramenti al datagen, § Ciclo v3+).

Esito della review "Data Distillation": i filtri di qualità standard (skip
scacco / bestmove tattica / score-cap / ply-min) e l'adjudication Syzygy sono
**già implementati** in `nnue/datagen.cpp`; l'unica aggiunta che paga è la
**deduplicazione**, da fare offline. Le altre idee valutate sono in fondo (§4).

---

## 1. Stato attuale (`nnue/datagen.cpp`)

| Meccanismo | Dove |
|---|---|
| Skip se in scacco / bestmove tattica (cattura, promo, ep) | `datagen.cpp:167-171` |
| Skip se \|score\| > 3000 cp / ply < 16 | `MAX_RECORD_SCORE_CP`, `MIN_RECORD_PLY` |
| Apertura 8-9 ply casuali, partita scartata se \|eval\| > 400 | `datagen.cpp:89-95,163-165` |
| Adjudication Syzygy ≤5 pezzi (no arrocco; decisivi solo hmc ≤ 60) | `datagen.cpp:128-147` |
| Adjudication win (±2500×4 ply) / draw (±10×8 ply da ply 80) | `datagen.cpp:176-182` |
| Un file per thread, una write bufferizzata per partita | `datagen.cpp:189-204` |

Throughput: ~1.000 pos/s × 32 B = **32 KB/s** — l'I/O non è un collo di
bottiglia (0,006% di un SSD), il costo è tutto nella search (~8 ms/mossa).

---

## 2. Dedup offline (il "cancello D")

Perché: i duplicati esatti ri-pesano le posizioni iper-frequenti (strutture
comuni, transposizioni) e riducono la diversità effettiva del dataset.

Come — **offline, non online**:

- **Chiave** = primi 24 byte del record (`occ` + `pcs`): la rappresentazione è
  stm-relative, quindi la dedup è già color-symmetric; l'en-passant non è nel
  formato, coerente con ciò che la rete vede.
- **Dove**: pass `dedup` (keep-first, log dei rimossi) nella pipeline
  pre-interleave, dove già gira il filtro re/occupancy del ciclo v2. Funziona
  sul merge multi-macchina, è esatto e riproducibile; su 366M record sono
  pochi minuti.
- **Non** in RAM durante la generazione: a 250M posizioni un `unordered_set`
  costa ~10 GB, un Bloom filter scarta posizioni buone sui falsi positivi, e
  nessuno dei due vede i duplicati cross-machine (laptop + fisso).

### TODO

- [x] **Fase 0 — MISURATO 2026-07-14** (`nnue/trainer/src/bin/datastats.rs`,
      FNV-1a 64 su chiave 24 B + sort): dataset v3 completo (33 file, 4
      macchine, **464.480.686 record, 0 azzerati, 0 code parziali**):
      **dup-rate 1,74%** (8,07M), top run 928; W/D/L 37,0/25,2/37,8;
      istogramma |score|: 11/9/15/20/20/25% (0-50/…/800-3000), >3000 = 0.
      (Nota: ply non nel record → dup-rate per ply non misurabile offline.)
- [x] **Fase 1 — DECISO: niente dedup** (1,74% ≪ 5%: ri-pesatura innocua).
- [~] **Fase 2 — non necessaria** per il v3; lo strip degli azzerati è
      comunque integrato in `datashuffle.rs` (merge = shuffle globale a shard,
      seed fisso — sostituisce l'interleave di bullet-utils, sparito dalla
      rev pinnata).
- [ ] (occasionale) contatori atomici di drop per motivo nel report del
      datagen — solo alla prossima modifica di `datagen.cpp`, non vale un
      rebuild dedicato con la run in corso.

---

## 3. Protocollo di validazione (se la dedup cambia il dataset in modo sostanziale)

Template = A/B nodi/mossa (`tuning/run_nodes_ab.sh`): due shakedown identiche
su dataset con/senza dedup → SPRT a TC 4+0.04 → si adotta solo su H1.
Se il dup-rate è basso non serve: dedup <2-3% è sotto la risoluzione dell'SPRT.

---

## 4. Idee valutate e scartate (2026-07-13 — non riproporre senza dati nuovi)

| Idea | Perché no |
|---|---|
| Filtro keep-only "alta divergenza search-vs-rete" (cancello A) | Segno invertito: alta divergenza ≈ tattica non risolta = etichette-rumore; keep-only distrugge la calibrazione. La pratica scarta il rumore, non lo seleziona |
| Harvest degli swing di eval (cancello C) | Lo score pre-swing era *sbagliato* (la search non vedeva la tattica): raccoglie sistematicamente etichette errate |
| "Oracolo" Syzygy mismatch (cancello B) | Già coperto dall'adjudication ≤5 pezzi; le fortezze stanno a 6-7 pezzi (TB ~70 GB); bulletformat non ha flag di priorità |
| Coda producer-consumer / ring buffer lock-free | I/O = 32 KB/s: problema inesistente; introdurrebbe l'unico punto di contesa oggi assente |
| Dedup online in RAM (set Zobrist / Bloom) | ~10 GB o falsi positivi; cieco cross-machine — l'offline è esatto e banale |
| Soglie dinamiche | Ogni knob = ~1 giorno di shakedown+SPRT; prima si valida il filtro statico |
| Rescoring / più nodi per etichette migliori | Già misurato: 12k = −85 ±23 vs 8k a parità di wall-clock — più posizioni > etichette migliori |

Variante salvabile del cancello A, solo come esperimento a protocollo §3:
*oversampling additivo* (duplicare, non selezionare) delle posizioni quiet ad
alta divergenza — preserva la distribuzione, enfatizza i gap posizionali veri.

---

## 5. Scoperta 2026-07-14: l'adjudication affama i bucket a basso materiale

La rete v3 (464M) valuta **KQvK = −13 cp**: l'adjudication Syzygy chiude le
partite appena entrano in ≤5 pezzi, quindi il bucket di output 0 (2-5 pezzi)
riceve ~zero esempi e resta ai pesi iniziali. (La ob v2 dava +1437 sulla stessa
posizione: fortuna dell'init, non training.) In partita il danno è mascherato
dal probe Syzygy, ma senza TB caricate l'eval dei finali minimi è rumore.

Fix candidato per il ciclo v4 (da misurare col protocollo §3): **quota di
partite seedate da finali** (8-16 pezzi, es. 5-10% dei game) — popola i bucket
bassi E genera esempi 6-7 pezzi fuori TB; in alternativa, registrare la
posizione terminale adjudicata con lo score TB come etichetta extra.
