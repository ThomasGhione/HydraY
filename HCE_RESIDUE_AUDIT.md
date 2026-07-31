# Audit residui HCE — intero progetto

**Data**: 2026-07-12 · **Baseline**: dev `d13d4de` (post prima pulizia: stallo=0, contempt su sola eval, via `incrementalMaterialDelta`) · **Solo analisi, nessuna modifica.**

Obiettivo: censire TUTTO il codice che esiste solo perché un tempo c'era l'evaluator handcrafted,
distinguendolo dalle tecniche di search che *sembrano* HCE ma sono legittime anche con NNUE.
Ogni file sorgente del progetto è stato esaminato (tabella di copertura in fondo).

Legenda verdetti:
- **RIMUOVERE** — residuo vero; behavior change → SPRT
- **TESTARE** — probabilmente obsoleto, ma il valore attuale è una domanda empirica → SPRT decide
- **REFACTOR** — residuo a costo zero (nessun behavior change; bench6 deve restare identico)
- **TENERE** — falso positivo: sembra HCE, non lo è
- **RENAME/DOC** — solo nome o commento stantio

---

## 1. Catena principale: `incrementalNonPawnMajorCount` e i suoi due unici consumer

> **✅ FATTO 2026-07-15** (merge `bf71fe1`): 1a+1b+1c testati in un unico SPRT vs dev@d5b7ba5
> (rete v3): **+17,57 ±10,13, LOS 99,97%, H1 @ 2592 partite** — guadagno vero, non solo
> non-regression. Nuovo baseline bench6 @ d12: **4.894.324**. Non re-aggiungere il gate
> pawn-endgame a RFP né la riga endgame dei margini futility senza un nuovo SPRT.

È la fotocopia della catena `incrementalMaterialDelta` appena demolita. Il campo incrementale
mantenuto in `doMove`/`undoMove` ([board.inl:152](board/board.inl#L152)) ha **esattamente due
consumer**, entrambi sospetti; il gate NMP (searcher.cpp:860) NON lo usa (fa un popcount locale
dei soli pezzi del lato a muovere). Se cadono entrambi, cade il campo → il piece-event hot path
resta con il solo accumulatore NNUE.

### 1a. `isPawnEndgameForPruning` — gate su RFP — **TESTARE (rimozione)**
- Dove: [searcher.cpp:818-819](engine/search/searcher.cpp#L818-L819) (calcolo),
  [searcher.cpp:913](engine/search/searcher.cpp#L913) (unico uso: gate RFP),
  [searcher.hpp:115](engine/search/searcher.hpp#L115) (campo in `SearchNodeState`).
- Cos'è: RFP disattivato quando `(esistono pedoni) && (nonPawnMajors totali ≤ 4)`.
- Perché è sospetto: RFP si fida della static eval per potare; il gate esiste perché la static
  eval **HCE** era notoriamente inaffidabile nei finali con pochi pezzi (re attivo, pedoni
  passati, fortezze). La NNUE — addestrata su centinaia di milioni di posizioni self-play con
  adjudication Syzygy — è proprio dove l'HCE era cieco. Stockfish non gata RFP così.
  Nota: NON è protezione zugzwang (quella serve a NMP, che ha il suo gate separato, vedi §4).
- Definizione peraltro strana: "pawn endgame" = *totale* majors ≤ 4 con pedoni — include
  posizioni con 2 pezzi per lato che pawn endgame non sono.
- Proposta: togliere il gate (RFP attivo ovunque tranne PV/check/root, come da precondizione).
- Verifica: bench6 (i nodi caleranno un filo) + SPRT `ELO0=0 ELO1=5` (è un *guadagno* atteso:
  più prune validi) o `[-3,3]` se lo si tratta da pulizia.
- Se passa: sparisce anche il campo da `SearchNodeState`.

### 1b. `isLateEndgame` + doppia riga futility/LMP — **TESTARE (collasso a riga singola)**
- Dove: [searcher.cpp:500-509](engine/search/searcher.cpp#L500-L509) (calcolo + uso),
  [search_constants.hpp:44-65](engine/search/search_constants.hpp#L44-L65)
  (`FUTILITY_MARGINS[2][7]`, `FUTILITY_EG_BASE/EG_STEP`, `LMP_BASE_THRESHOLDS[.][2][.]`),
  [searcher.cpp:47-62](engine/search/searcher.cpp#L47-L62) (`rebuildSearchDerivedTables`),
  [uci.cpp:95-96](uci/uci.cpp#L95-L96) (spin `FUTILITY_EG_BASE`, `FUTILITY_EG_STEP`).
- Cos'è: margini futility e soglie LMP diversi quando `nonPawnMajors totali ≤ 5`.
- Perché è sospetto: la biforcazione per fase fu progettata e tarata (giugno, tree HCE) attorno
  alle caratteristiche dell'eval HCE, che in endgame era smooth/material-driven e giustificava
  margini più stretti. I motori di riferimento usano un margine unico per depth; la volatilità
  dell'eval NNUE non segue il confine "≤5 majors". In ogni caso quelle costanti vanno ritarate
  su NNUE (SMAC3 #9 è già in piano): collassare PRIMA di ritarare dimezza i parametri.
- Proposta: usare la sola riga "mid" (margini `260·d`, LMP base standard); se SPRT regge,
  eliminare riga EG, i 2 spin UCI e la dimensione `[lateEndgame]` di LMP.
- Verifica: bench6 + SPRT `[-3,3]`.
- Rischio: medio — è l'unico candidato dove l'ipotesi "HCE-driven" è indiretta; se SPRT boccia,
  tenere la struttura e lasciarla ritarare a SMAC3.

### 1c. Demolizione finale — **REFACTOR (solo dopo 1a+1b)**
Se entrambi passano: via `incrementalNonPawnMajorCount`
([board.hpp:213](board/board.hpp#L213), [board.hpp:295](board/board.hpp#L295),
[board.inl:43](board/board.inl#L43), [board.inl:78](board/board.inl#L78),
[board.inl:152](board/board.inl#L152)) e l'`if constexpr` che lo mantiene in
`updatePieceTypeBB`. A quel punto bench6 deve restare **identico** al valore post-1a+1b,
`nnue-selftest` obbligatorio (si toccano i piece hooks).

---

## 2. Contempt sulle ripetizioni — risposta al dubbio + **TESTARE (opzionale)**

Domanda posta: «la patta per ripetizione viene già considerata da NNUE?» — **No, e non può**:
la NNUE è una funzione *statica* di una singola posizione, non ha input di storia della partita.
Una posizione che sta per ripetersi la terza volta ha la stessa eval della prima occorrenza.
Il **rilevamento** (2-fold/3-fold/50 mosse/materiale insufficiente,
[searcher.cpp:174-217](engine/search/searcher.cpp#L174-L217)) deve restare in search: non è
residuo HCE, è obbligo di ogni motore.

Quello che invece È testabile è la **policy** sul 3-fold:
- Dove: `repetitionDrawScore` [searcher.cpp:220-233](engine/search/searcher.cpp#L220-L233),
  `REPETITION_CONTEMPT` [search_constants.hpp:135](engine/search/search_constants.hpp#L135),
  `REPETITION_DRAW_ADVANTAGE_THRESHOLD` [searcher.cpp:20](engine/search/searcher.cpp#L20).
- Oggi: 3-fold = ∓80cp se |eval NNUE| > 50, altrimenti 0. Stockfish e i motori moderni usano 0
  secco (al più un dither ±1 per rompere le simmetrie di ricerca).
- Onestà: questo NON è un residuo HCE in senso stretto (il contempt è una policy anti-draw
  legittima), ma è l'ultimo pezzo di scoring "hand-crafted" rimasto in search. Se si vuole
  l'estremo rigore del principio "la conoscenza sta nella rete": SPRT `[-3,3]` con
  `repetitionDrawScore → 0`. Se passa: −20 righe, via 2 costanti e una chiamata a
  `Evaluator::evaluate` sui nodi 3-fold.
- Rischio: a TC corti il contempt evita patte da posizioni vincenti; possibile piccolo Elo
  negativo. È il candidato con più probabilità di essere *bocciato* dall'SPRT.

---

## 3. `eval_constants.hpp`: costante morta + constexpr mancanti — **REFACTOR**

> **✅ FATTO 2026-07-15** (`dfa1db5`, insieme a §5): piece values/PIECE_VALUES/MVV_TABLE
> constexpr, MATE_SCORE rimosso, commenti sistemati. bench6 identico (4.894.324).

- **`MATE_SCORE` è morto**: [eval_constants.hpp:22](engine/eval_constants.hpp#L22) — l'unico
  riferimento nel progetto è un *commento* in [syzygy.hpp:34](engine/syzygy/syzygy.hpp#L34)
  (il codice vero usa `POS_INF`/`NEG_INF`/`MATE_BOUND`). Rimuovere entrambi.
- **I piece value non sono più mutabili a runtime**: con le spin `STALEMATE_*` rimosse
  (d13d4de), nessuno scrive più `PAWN_VALUE…KING_VALUE`, `PIECE_VALUES[]`, `MVV_TABLE[]`.
  Sono ancora `inline int32_t` (retaggio delle campagne di tuning eval HCE) → farli
  `constexpr`. `PIECE_VALUES`/`MVV_TABLE` sono letti in SEE/ordering (Tier 2 hot): const
  propagation e niente reload dalla memoria. Cascata: il commento stale
  [searcher.cpp:18-19](engine/search/searcher.cpp#L18-L19) («PAWN_VALUE is a runtime-mutable
  eval global») muore e `REPETITION_DRAW_ADVANTAGE_THRESHOLD` diventa `constexpr`.
- Verifica: bench6 **identico** (refactor puro) + `make prod` pulito.
- Nota: i *valori* (100/344/359/502/960) furono tarati per l'eval HCE ma oggi servono solo a
  SEE/MVV/delta-pruning, dove valori simili sono standard ovunque. Ritararli è micro-tuning a
  bassa priorità (eventualmente spin temporanei in una campagna SMAC3 dedicata), non pulizia.

---

## 4. Falsi positivi — TENERE (e perché)

| Cosa | Dove | Perché resta |
|---|---|---|
| Delta pruning qsearch con `PIECE_VALUES` | [sorter.cpp:273-285](engine/sort/sorter.cpp#L273-L285), [searcher.cpp:1084-1115](engine/search/searcher.cpp#L1084-L1115) | Upper bound *materiale* sul guadagno di una cattura: è la definizione della tecnica, serve un valore per pezzo. Standard in ogni motore NNUE. |
| SEE (`staticExchangeEvaluation`) | sorter.cpp | Ordering + pruning delle catture: opera per definizione sui valori dei pezzi. |
| `MVV_TABLE` | sorter.cpp | Most-Valuable-Victim: idem. |
| Gate NMP `nonPawnMajors >= 2` (lato a muovere) | [searcher.cpp:860-903](engine/search/searcher.cpp#L860-L903) | Protezione **zugzwang**: correttezza della null-move observation, indipendente dall'evaluator. Presente in ogni motore. (Popcount locale, non usa il campo incrementale.) |
| Correction history pawn/minor/major | [searcher.cpp:66-89](engine/search/searcher.cpp#L66-L89), searchruntime.hpp | Tecnica *post*-NNUE (corregge la static eval con i residui osservati): non è conoscenza handcrafted, è apprendimento online. |
| Rilevamento patte (2-fold/3-fold/50m/insuff.) | searcher.cpp:174-217, boardapi | La rete non vede la storia della partita (vedi §2). |
| `hasInsufficientMaterialDraw` | [board.cpp:349](board/board.cpp#L349) | Regola FIDE, non euristica. |
| Syzygy (probe WDL/root, adjudication datagen) | engine/syzygy/, searcher.cpp | Conoscenza esatta, ortogonale all'evaluator. |
| Check extension, IIR, SE, LMR, history/killer/counter | searcher.cpp, sorter | Search puro, nessuna dipendenza dall'evaluator. |
| `evalStack`/`improving` | searcher.cpp:846-857 | Confronta la static eval con se stessa 2 ply prima: funziona con qualunque evaluator. |

## 5. Rename / commenti stantii — **RENAME/DOC** (zero rischio)

> **✅ FATTO 2026-07-15** (`dfa1db5`): QSEARCH_STANDPAT_*, via i riferimenti YBWC e gli
> altri commenti stantii della tabella.

| Dove | Problema |
|---|---|
| [search_constants.hpp:115-118](engine/search/search_constants.hpp#L115-L118) `QSEARCH_MATERIAL_BAD/WORSE(_DELTA)` | Nome fuorviante: confrontano lo **standPat NNUE**, non il materiale. Rinominare tipo `QSEARCH_STANDPAT_BAD/WORSE`. |
| [searcher.cpp:748-749](engine/search/searcher.cpp#L748-L749) | «…is scored by static material» — stale: oggi il nodo orizzonte è scored da eval NNUE/0. |
| [searcher.cpp:18-19](engine/search/searcher.cpp#L18-L19) | «PAWN_VALUE is a runtime-mutable eval global» — falso da d13d4de (vedi §3). |
| [movepicker.hpp:97](engine/sort/movepicker.hpp#L97) e [searchruntime.hpp:26-28](engine/search/searchruntime.hpp#L26-L28) | Riferimenti a **YBWC**, sostituito da Lazy SMP nel 2026-07-04. |
| [time_manager.cpp:68-71](engine/time/time_manager.cpp#L68-L71) | «the engine has no opening book» — falso: komodo.bin caricato e `Opening=true` di default (vedi §7). |
| `search_constants.hpp:8-9` | «…the same way eval_constants.hpp centralizes evaluation weights» — eval_constants non contiene più pesi eval. |

## 6. Test suite: rotta da riferimenti HCE — **RIMUOVERE/RISCRIVERE**

> **✅ FATTO 2026-07-15** (`88fc82d`): potati i 4 file engine/test bit-rotted (−753 righe),
> mainTest riparato (include HCE, attivazione rete embedded, posizione threefold spostata
> dal bucket-0 affamato, contratto root-draw aggiornato), testBoard su API attuali.
> **`make test` è di nuovo VERDE** (393 assert / 12 test) — aggiornare la nota
> "make test is broken" in memoria/doc.

`make test` non compila, e la causa primaria è proprio l'HCE rimosso:

| File | Problema | Tipo |
|---|---|---|
| [tests/mainTest.cpp:4](tests/mainTest.cpp#L4) | `#include "../engine/eval/evaluator.hpp"` — directory cancellata con l'HCE | HCE |
| [testEngine.cpp:54](engine/test/testEngine.cpp#L54) | `Engine::getMaterialDelta()` — API rimossa; il test usava il material delta come guardrail anti-blunder (concetto HCE-era, oggi coperto da `make sacrifice`) | HCE |
| [EndingGame.cpp:89](engine/test/EndingGame.cpp#L89), [criticalPositionEngine.cpp:75](engine/test/criticalPositionEngine.cpp#L75) (e altri 8 siti) | `e.evaluate(e.board)` — `Engine::evaluate` non esiste più | HCE |
| board/test/testBoard.cpp | `getCurrentFen`, `get(std::string)` — API Board cambiate | drift, non HCE |

Opzioni: (a) riparare i test sostituendo le API morte (`Evaluator::evaluate`, via i guardrail
material-delta); (b) potare i test HCE-era e tenere solo board/movegen/mate — decisione utente.
Finché non si fa, il protocollo resta «make + bench, MAI make test» (già nelle note di progetto).

## 7. Emerso durante l'audit, NON HCE (nota a margine, decisioni separate)

- **Opening book polyglot attivo di default**: `engine/komodo.bin` (9,2 MB) caricato in ogni
  costruzione dell'Engine ([engine.cpp:70](engine/engine.cpp#L70)), `Opening=true`
  ([engine.hpp:86](engine/engine.hpp#L86)), opzioni UCI `BookFile`/`Opening`. Negli SPRT è
  simmetrico (fair), ma il motore può giocare mosse di libro *dopo* l'opening imposta dal
  tester se rientra in book. Da valutare: default off in UCI (i tester/lichess-bot gestiscono
  già le aperture) o rimozione della feature. ~730 righe (opening/ + polyglot_keys).
- **Pondering**: thread + logica dedicata in engine.cpp; usato solo in terminal mode? Verifica
  d'uso rinviata (fuori scope HCE).

---

## 7-bis. Secondo passaggio 2026-07-15 — nessun nuovo residuo

Ri-sweep mirato dopo il merge di §1/§3/§5/§6 (pattern: termini eval classici, penalty/bonus/
weight fuori dalle history, duplicati di costanti): **nessun residuo HCE nuovo**. Ciò che
resta di "DNA HCE" nel motore è, per intero: §2 (policy contempt — decisione utente),
§7 (opening book — decisione utente, non-HCE), i *numeri* dei piece value (344/359/502/960,
tarati per l'HCE ma oggi solo SEE/MVV/delta — ritararli è micro-tuning SMAC3, non pulizia)
e le costanti di search tarate a giugno sul tree HCE → la retune SMAC3 #9 è ora ancora più
motivata (rete nuova + RFP ovunque + futility a riga singola = albero molto diverso).
Note minori senza azione: `Engine::DEFAULTDEPTH` alias ridondante di `DEFAULT_DEPTH`;
`QSEARCH_DELTAMARGIN_MIN = 960 // == QUEEN_VALUE` resta letterale (un include in più non
vale la sostituzione).

## 8. Piano di esecuzione proposto (quando si deciderà di procedere)

Ordine pensato per massimizzare l'informazione per SPRT speso e non mischiare refactor e
behavior change:

1. **Batch refactor a rischio zero** (§3 + §5): constexpr, MATE_SCORE, rename, commenti.
   Verifica: `make prod` pulito, bench6 **identico** (5.230.424), selftest. Niente SPRT.
2. **§1a — via il gate pawn-endgame da RFP.** bench6 (annotare nuovo valore) + SPRT.
3. **§1b — collasso futility/LMP a riga singola.** bench6 + SPRT `[-3,3]`.
4. **§1c — demolizione `incrementalNonPawnMajorCount`** (solo se 2+3 passano).
   bench6 identico al post-3 + `nnue-selftest` + `make sacrifice`.
5. **§2 — contempt→0** (opzionale, singolo SPRT `[-3,3]`; aspettarsi possibile bocciatura).
6. **§6 — test suite**: riparazione o potatura (nessun impatto Elo; sblocca `make test`).

Nota datagen: come per la pulizia precedente, i punti 2/3/5 cambiano leggermente la search del
labeler. Mix pre/post nel dataset v3 = non problematico (stessa rete etichettatrice); nessun
motivo di buttare dati.

Dopo l'intero piano, `SearchNodeState` si riduce, `search_constants.hpp` perde ~15 righe di
tabelle/generatori, 2 spin UCI in meno, e la coppia doMove/undoMove mantiene: squares,
bitboards, zobrist, repetition ring, accumulatore NNUE — e basta.

---

## Appendice: copertura dell'audit (file per file)

| Area | File | Esito |
|---|---|---|
| board/ | board.hpp, board.cpp, board.inl, boardapi.inl, boardapi.cpp, fen.cpp, piece.hpp, coords.hpp, magic_numbers.hpp | Pulito salvo §1c (campo incrementale). Magic/attacchi/FEN: nessun residuo. |
| engine/ | engine.hpp, engine.cpp, evaluator.{hpp,cpp}, eval_constants.hpp, movelist.hpp | §3; evaluator = seam NNUE corretto; engine.cpp: book §7, updateGameResult pulito. |
| engine/search/ | searcher.cpp, searcher.hpp, search_constants.hpp, searchruntime.hpp | §1a, §1b, §2, §5. Aspiration/ID/root/qsearch/TT-interplay: puliti. |
| engine/sort/ | sorter.cpp, sorter.hpp, movepicker.hpp, move_generator.{cpp,hpp} | Tenere (SEE/MVV/delta = §4); YBWC doc §5. |
| engine/time/ | time_manager.{cpp,hpp} | Pulito; commento book §5. Nessuna allocazione tempo material-based. |
| engine/opening/ | opening_book.cpp, polyglot_keys.hpp | Non HCE; vedi §7. |
| engine/syzygy/ | syzygy.{cpp,hpp} | Pulito; commento MATE_SCORE §3. |
| tt/ | tt.hpp, zobrist.hpp | Pulito (nessun accoppiamento con l'eval). |
| uci/ | uci.cpp, uci.hpp | Spin rimaste = tutte search vere; opzioni book §7. |
| driver/ | driver.{cpp,hpp} | Pulito (display/save-load; usa solo GameResult). |
| nnue/ | nnue.cpp, network.hpp, accumulator.hpp, embedded.cpp, datagen.cpp, selftest.cpp, bulletformat.hpp | Pulito (i filtri datagen usano l'eval per design). |
| root | main.cpp, debug.hpp, ascii_utils.hpp, makefile | Puliti; makefile: target `test` rotto per §6. |
| tests | tests/*, engine/test/*, board/test/*, driver/test (vuota) | §6. perf/sacrifice/nodebench compilano e girano. |
| tuning/ | groups/*.json, script | Nessun riferimento alle spin rimosse (STALEMATE_* non era nei gruppi). |
