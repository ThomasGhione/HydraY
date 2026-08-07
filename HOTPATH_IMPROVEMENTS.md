# Hot-Path Improvements — analisi 2026-07-03

Analisi completa dell'hot path (searcher, sorter, movepicker, move_generator, TT, zobrist,
board doMove/undoMove/inCheck), evaluator escluso (futuro NNUE). Ordinata per criticità.

**Metodo di verifica**
- *Behavior-preserving*: node-count byte-identico via UCI (`Threads 1`, `Opening false`,
  `ucinewgame` + `go depth 12` su 6 posizioni miste). `make perf` misura TEMPI, non nodi.
- *Behavioral*: SPRT via `tuning/run_sprt.sh` (non-regression `ELO0=-3 ELO1=3`, gain `[0,5]`).
- Stime Elo da NPS: +10% velocità ≈ +7–10 Elo a TC fisso.

---

## 🔴 Criticità ALTA — problemi di performance reali nell'hot path

- [x] **1. Guardia `halfMoveClock >= 4` sul check ripetizioni** — Elo atteso: **+3–8** (NPS)
  ✅ 2026-07-03, node-identico. Con #2: +3.45% NPS (A/B interleaved 4×).
  `checkDrawTerminalConditions` chiama `countRepetitions()` (scan lineare O(historySize))
  a OGNI nodo, qsearch inclusa (50–80% dei nodi). Una ripetizione richiede
  `halfMoveClock >= 4`; in qsearch ogni mossa è cattura/promozione → clock quasi sempre 0.
  Aggiungere la guardia salta lo scan per quasi tutta la qsearch. Bonus endgame: scan
  all'indietro con early-exit a count 2/3. Node-identico per costruzione.

- [x] **2. Riposizionare il prefetch TT dopo `doMove`** — Elo atteso: **+2–5** (NPS)
  ✅ 2026-07-03, node-identico (vedi #1 per l'NPS combinato).
  `handleSearchPrelude` fa `prefetch(hashKey)` seguito subito da `probe(hashKey)` sulla
  stessa chiave: latenza zero nascosta (le note di progetto dicono "2 nodes ahead" ma il codice non lo fa).
  Spostare in `searchMoves` e nel loop qsearch subito dopo `b.doMove()`:
  `tt->prefetch(b.getHash())` — la latenza si nasconde dietro enterNode+draw-check+eval del figlio.
  Node-identico.

- [x] **3. Unified TT probe: 1 snapshot per nodo invece di 4** — Elo atteso: **+3–6**, −40 LOC
  ✅ 2026-07-03 (bb9f5f8), node-identico. NPS **neutro** (−0.24% interleaved: i re-probe
  erano L1-hot) — tenuto per il valore strutturale (−40 LOC, sorter senza TT, un solo
  punto di lettura da estendere col campo static-eval per NNUE).
  Per nodo il bucket viene letto fino a 4 volte: `probe()` (prelude), `probeSE()` (static-eval
  tightening), `probeSE()`+`probeMove()` (gate singular), `probeMove()` (sortLegalMoves).
  Un unico `probeEntry(key) → {found, score, flag, depth, move}` a inizio nodo serve tutti;
  la hash move viene passata a `sortLegalMoves` come parametro (muore `outHashMoveIsLegal`).
  In tt.hpp `probe/probeSE/probeMove` collassano in una funzione. Node-identico (single-thread).
  Nota: qui si aggiungerà il campo static-eval nel payload quando arriverà NNUE.

- [x] **4. Gate su `tryStalemateScore` nei cutoff RFP/NMP** — ❌ **SCARTATO** 2026-07-03
  Node-identico ma NPS −2.8% su 9 round A/B (l'early-out di hasAnyLegalMove è già economico;
  delta dominato dal code-layout). Guadagno non dimostrabile → revert, niente 5 righe in più.
  Ogni fail-high RFP/NMP paga `hasAnyLegalMove()`. Stallo con eval ≥ beta+margine e
  materiale non-pedone è ~impossibile (RFP già gated `!isPawnEndgameForPruning`).
  Gate su materiale basso (es. `nonPawnMajorCount <= 4`) o rimozione dal path RFP.
  Behavioral → SPRT non-regression.

## 🟠 Criticità MEDIA — leve Elo non ancora provate (SPRT)

- [x] **5. Riabilitare TT-write nei *discendenti* del probe singolare** — ✅ **KEPT** 2026-07-04
  SPRT [0,5]: **+9.4 ±16.1 @ 1258, CFS 92** (precedente dd57d1f) + razionale Lazy-SMP.
  Nota: passare `allowTTWrite` al call-site era inerte (il nodo excluded azzerava il flag
  e lo propagava); fix vero = gate `!hasExcludedMove` sul solo store del nodo.
  La SE passa `allowTTWrite=false` che si propaga a tutto il sottoalbero; solo il nodo con
  `excludedMove` ha la chiave "sbagliata" (soppressione locale già presente via
  `hasExcludedMove`). Passare `true` alla ricorsione del probe SE.

- [x] **6. Negative extension nel blocco singular** — ❌ **BOCCIATO** 2026-07-03
  SPRT [0,5]: **−10.9 ±19.4 @ 860 partite, LLR −0.56, LOS ~8%** — killed early, revert.
  Implementazione standard (`ttScore>=beta → ext −1`); conferma il meta-pattern: aggiungere
  termini di riduzione a questo search tunato non paga. Non ritentare senza gating diverso.
  Il gruppo SE è stato la miniera del ciclo 2026-06 (SE_BETA_MARGIN +16.6). Ramo mancante
  standard: quando `seScore >= seBeta` e `ttScore >= beta`, ridurre di 1 il primo figlio
  invece di estendere. Unica leva SE mai provata.

- [x] **7. Store bound-only sul stand-pat fail-high in qsearch** — ✅ **KEPT** 2026-07-03
  SPRT [0,5]: **+22.6 ±27.9 @ 400, CFS ~99**, accettato early (decisione utente).
  Il return `standPat >= beta` esce PRIMA dello store. `store(hash, 0, standPat, LOWERBOUND)`
  senza mossa (bestMove=0 preserva la mossa esistente) evita la move-pollution che uccise
  l'esperimento qsearch-TT-con-mosse (−30 Elo).

- [x] **8. `improving` con fallback a ply−4** — ⚪ **NEUTRO, scartato** 2026-07-03
  SPRT [0,5]: −1.9 ±18.4 @ 900. Nessun segnale → revert (niente righe extra per zero Elo).
  Nota: 2 run interrotti da "engine not responsive" (~1/900 partite, entrambi i lati,
  solo stasera) — probabile carico macchina; da monitorare.
  Quando `evalStack[ply-2] == NEG_INF` (in scacco 2 ply fa) `improving` è sempre false;
  fallback standard a ply−4. ~3 LOC.

- [x] **9. Campagna SMAC3 sulle costanti di search** — RESA: **+17,74** (2026-08-03)
  Tutti i margini vinti a giugno (FUTILITY_MARGINS, LMP, HISTORY_PRUNE, SEE_CAPTURE, RFP,
  NMP_EVAL_*, PROBCUT_*, SE_*) sono guessed, not tuned.
  - [x] **Esposizione UCI FATTA 2026-07-05**: 16 spin (pattern eval_constants: globali
        mutabili + `rebuildSearchDerivedTables()` per LMR/futility/LMP derivate; tabelle
        parametrizzate da generatori che riproducono ESATTAMENTE i valori attuali —
        node-identico 5.782.300 ai default, roundtrip set/restore verificato).
        Gruppi pronti: `tuning/groups/search_pruning.json` (8 param margini) e
        `search_shape.json` (8 param futility/LMP/LMR/SE).
  - [x] **RUN tuning FATTO 2026-08-03** — risultato: **+17,74 ±8,67** (SPRT 3294 partite,
        LOS 100%, H1 raggiunto da solo), commit a1dcbaf. I gruppi sono stati ridivisi in
        tre insiemi disgiunti perché quelli originali si sovrapponevano e `search_shape`
        tarava due costanti cancellate a luglio (`FUTILITY_EG_*`), bruciando due
        dimensioni su rumore puro.
        - `search_evalmargins` (RFP, NMP_EVAL_*, FUTILITY_MID_STEP, PROBCUT) →
          **+17,74**. Sono le uniche costanti che confrontano una eval statica con una
          soglia in centipawn, cioè le uniche starate dal passaggio a NNUE.
        - `search_shape` (LMP ×2, LMR_C, SE ×2) → **−1,3 ±4,6 dopo 153 iterazioni,
          NESSUN margine**. Girato DOPO aver applicato i margini nuovi, quindi vale anche
          con le potature correnti: la geometria dell'albero non dipende dalla scala dei
          punteggi ed è sopravvissuta intatta alla rimozione dell'HCE. **Non ritarare.**
        - `search_ordering` (SEE_CAPTURE, HISTORY_PRUNE ×3) → **+1,0 ±3,7 dopo 128
          iterazioni, NESSUN margine**. L'argmax è saltato fra due regioni lontane
          (SEE 74/HP_D3 −11350 e SEE 35/HP_D3 −4000) valutandole uguali: firma di
          superficie piatta, più netta dell'Elo stesso. **Non ritarare.**
        **Conclusione**: si erano starate SOLO le costanti che confrontano una eval
        statica con una soglia in cp. Geometria dell'albero e statistiche di
        ordinamento non dipendono dalla scala dei punteggi e sono sopravvissute
        intatte alla rimozione dell'HCE. La campagna è CHIUSA.
        ⚠️ **Il tuner girava a profondità fissa 8** (`base_config.json`, dall'era HCE): a
        profondità fissa una costante di pruning può solo essere premiata se pota di meno,
        quindi la prima campagna è finita nell'angolo della scatola dichiarando +40 falsi.
        Corretto in controllo di tempo. Profondità fissa va bene solo per i **pesi di
        valutazione**, che non barattano nodi contro profondità.

- [x] **10. TB probe dopo il TT probe** — ✅ **FATTO** 2026-07-04
  Node-identico senza TB (blocco inerte); con TB attive = meno probeWDL a parità di cutoff.
  Il probe Syzygy (mmap) avviene prima del TT cutoff: in TB-range ogni nodo lo paga anche
  quando il TT avrebbe tagliato. Spostare il blocco dopo il prelude.

- [x] **11. Lazy SMP al posto di YBWC** — ✅ **FATTO** 2026-07-04
  YBWC root rimosso (~150 LOC); helper threads con SearchRuntime privati persistenti
  (HelperSlot), TT condivisa, odd helpers partono a depth 2. TTD depth-13 4-pos:
  Lazy 4T 17.6s vs YBWC 4T 19.2s (~9% più veloce) vs 1T 27.6s (1.57×).
  Albero 1T −21% nodi (root loop PVS pulito con euristiche). SPRT 1T non-regression
  [−3,3]: +~8 Elo trend, CFS 87 @ 1835 game → accettato.

## 🟡 Codice morto / duplicato — zero Elo, −LOC (node-count identico)

- [x] **12. Fondere `generateLegalEvasionsFor` in `generateLegalMovesFor`** — **−60 LOC**
  ✅ 2026-07-03 (9928ca3), node-identico, −47 LOC netti via CheckContext.
  Il generatore completo gestisce GIÀ lo scacco (evasionMask, double-check early-return,
  castling gated `!inCheck`); la versione evasioni è lo stesso codice meno il castling.
  Unificare con un parametro `CheckInfo {known, inCheck, doubleCheck}`.

- [x] **13. Rimuovere parametro `useTT` (sempre true) + `useHashMove`** — **−15 LOC**
  ✅ 2026-07-03, node-identico.
  Nessun call-site esterno passa mai false: `canUseTT` collassa in
  `runtime.transpositionTable != nullptr`. Idem `useHashMove` di sortLegalMoves
  (il null-check interno c'è già).

- [x] **14. Check re-legalità del re in qsearch morto** — **−4 LOC**
  ✅ 2026-07-03, node-identico.
  searcher.cpp ~1084: le mosse vengono da `generateTacticalMovesFor` che filtra GIÀ le
  catture del re con `isLegalPseudoMove`. Il re-check è sempre true.

- [x] **15. Campo `MovePicker::hashMoveIsLegal` mai letto** — **−6 LOC**
  ✅ 2026-07-03, node-identico.
  Scritto in sortLegalMoves, nessun reader (il searcher usa `outHashMoveIsLegal`).

- [x] **16. `MovePicker::size` duplica `moves.size`** — **−8 LOC**
  ✅ 2026-07-03, node-identico.
  Due campi da tenere in sync in 3 punti. Usare solo `moves.size`.

- [x] **17. Wrapper `Sorter::givesCheckFast` 1:1** — **−5 LOC**
  ✅ 2026-07-03, node-identico.
  Forwarding puro a `givesCheckAfterQuietMoveFast`: rendere pubblica quella vera.

- [x] **18. `betaOrig` in qsearch** — **−2 LOC**
  ✅ 2026-07-03, node-identico.
  `beta` non è mai modificato in qsearch; la copia è inutile.

- [x] **19. Passare `inCheck` a `sortLegalMoves`** — assorbito da #24 (il ramo che lo usava è stato rimosso)
  sorter.cpp ricalcola `b.inCheck()` per nodo solo per il ramo opening-king;
  il chiamante lo conosce già. (Superato da #24 se si rimuove l'euristica.)

- [x] **20. `checkersTo(kingSq)` unificato** — **−20 LOC**, +NPS nei nodi in scacco
  ✅ 2026-07-03 (761d5b1), node-identico, −43 LOC netti. `isDoubleCheck` e
  `computeCheckEvasionMasks` eliminati; CheckContext = {known, checkers}.
  Oggi 3 scan di attacco per nodo in scacco: `inCheck()` → `isDoubleCheck()` →
  `computeCheckEvasionMasks` (ricalcola i checkers). Un unico bitboard risponde a tutte
  e tre (≠0, >1 bit, maschera evasioni).

- [x] **21. Conversione promo-char triplicata** — ✅ **FATTO** 2026-07-05, node-identico
  `Move.promotionPiece: char` → `Move.promotionType: uint8_t` (piece-type Board, 0 = none).
  Eliminate le 3 tabelle (`normalizePromotionChoice`+`promotedPieceFromChoice`,
  `Sorter::promotionPieceType`, `TT::promoCodeFromChar/CharFromCode`); conversione char
  solo ai boundary (`Move::promotionTypeFromChar`/`promotionChar`, static_assert vs
  piece_id). doMove promo path: niente più compare di char, `promo | color` diretto.
  −14 LOC nette, node-identico 5.782.300.

- [x] **22. Ponder-move da TT duplicato** — ✅ **FATTO** 2026-07-04
  `TT::probeDecodedMove()` unifica i 3 siti probe+decode (engine.cpp, uci.cpp, buildPvFromTT).

- [x] **23. Minori** — ✅ 2026-07-03 (1ad2d0f), node-identico
  - [x] `LMR_MAX_MOVES`/`MAX_MOVES`: cross-reference nel commento (unificarli costerebbe
        include-churn movelist↔search_constants — non vale la pena)
  - [x] `LMP_THRESHOLDS[..][6]`: entry [5] mai indicizzata → array [5]
  - [x] Commento stale su FUTILITY_MARGINS corretto (1..6)
  - [x] `TT::probe` const (poi assorbito da probeEntry in #3)
  - [x] hydray.md: claim "prefetch 2 nodes ahead" corretto insieme a #2 (2026-07-03)

## 🟢 Candidati "remove-to-gain" (SPRT; pattern vincente 3/3 a giugno: +19, +10.9, +8.8)

- [x] **24. Opening-king penalty + castling bonus nell'ordering** — ✅ **RIMOSSO** 2026-07-03
  Commit e0/[REMOVE], −15 LOC. SPRT [−3,3]: +43 ±51 @ 138, trend netto positivo, accettato
  early (decisione utente). Elimina anche il `b.inCheck()` per nodo nel sorter (#19 assorbito).

- [x] **25. SEE cache 1 MB thread_local** — ✅ **RIMOSSA** 2026-07-03
  Node-identica (cache esatta), **+1.76% NPS** (5 round interleaved), −22 LOC, −1 MiB/thread.
  64K entry × 16 B per thread. Con la lazy SEE le chiamate sono crollate: la cache
  potrebbe non ripagare la cache-pressure. Se neutra: −25 LOC e 1 MB/thread in meno
  (rilevante per Lazy SMP).

- [x] **26. Layout killer `[2][MAX_PLY]` → `[MAX_PLY][2]`** — ✅ **FATTO** 2026-07-03
  Node-identico; **+5.05% NPS cumulato con #25** (5 round interleaved).
  killer0/killer1 dello stesso ply oggi distano 192 B (2 cache line). Micro, node-identico.

## 🔍 Indagini aperte

- [ ] **Stalli "engine not responsive" negli SPRT** (~1 ogni 200–900 partite dal 2026-07-03
  sera): colpiscono ENTRAMBI i lati, anche binari pre-Lazy-SMP, a concurrency 2–4.
  Iniziati coi run dove entrambi i lati includono #7 (stand-pat store) — correlazione
  non dimostrata; alternative: carico macchina, pondering interno / SearchApiMutexGuard.
  Mitigato con `-recover` in run_sprt.sh (partita persa al lato stallato). Da investigare:
  riprodurre e attaccare gdb al processo bloccato.

## ⛔ Da NON rifare (già testato e bocciato — vedi memoria sessioni 2026-06)

staged movegen (inerentemente non node-neutral) · razoring (×2) · cutNode / LMR-do-deeper ·
LMR da moveIndex 3 · contHist multi-ply nell'ordering · contHist in LMR/pruning ·
qsearch TT **con mosse** · estensione di scacco a tutte le depth · lazy eval ·
rimozione SEE per-quiet nell'ordering (load-bearing, +80% nodi senza).

---

## Ordine di esecuzione consigliato

1. **Behavior-preserving node-identici**: #14, #18, #15, #16, #17 → #13 → #1, #2 → #23 → #3 → #12, #20
2. **SPRT one-shot economici**: #4, #5, #6, #7, #8, #24, #25
3. **SMAC3 search constants** (#9, richiede esposizione UCI una-tantum)
4. **Lazy SMP** (#11, quando c'è l'hardware)

**Baseline node-count (6 posizioni, depth 12, Threads 1, Opening off): 7.922.716**
(startpos 1.257.726 · kiwipete 325.684 · endgame 17.810 · midgame 3.396.556 ·
tactical 869.242 · open 2.055.698 — driver: `nodebench.py`, vedi memoria tooling-nodebench)

**Log:** 2026-07-03 — batch #14/#18/#15/#16/#17 (d1d1dfe), #13 (8ba81c1), #1/#2 (0e3303c),
#23 (1ad2d0f), #3 (bb9f5f8), #12 (9928ca3): tutti node-identici (7.922.716 dopo ogni commit).
NPS: **+3.45%** dai batch 1–3 (A/B interleaved); #3 neutro (−0.24%, tenuto per struttura/LOC).
2026-07-04 — **#11 Lazy SMP committato**: SPRT 1T non-regression CFS 87 @ 1835 game,
TTD 4T −9% vs YBWC, baseline node-count 1T nuova: **5.782.300**.
2026-07-05 — **#21 promo-as-piece-type committato**: node-identico (5.782.300), −14 LOC,
promozioni verificate end-to-end (UCI parse + bestmove + book/syzygy boundary).
Resta: #9 (SMAC3, serve esposizione UCI una-tantum; il RUN va fatto a laptop libero —
il datagen NNUE occupa 3 core fino a ~2026-07-09, misure a tempo falsate fino ad allora).
