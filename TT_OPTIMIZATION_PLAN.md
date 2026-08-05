# Transposition Table — analisi e piano di ottimizzazione

Analisi di `tt/tt.hpp` fatta il 2026-08-04 su profilo `perf` a **cicli** (non a
istruzioni: la differenza è decisiva, vedi §0).

## STATO 2026-08-05 — §1, §3, §4 FATTI

| § | esito | misura |
|---|---|---|
| §3 static eval nel payload | ✅ `986b93d` | **+2,3%** a d14, albero identico |
| §1+§4 XOR-lockless | ✅ `7b289ef` | **+1,5%** medio / +2,2% sul minimo, −121 righe |
| §7 huge page | ⏳ serve `sudo sysctl -w vm.nr_hugepages=64` | non eseguibile senza sudo |
| §2 chiave a 16 bit | ⏳ non fatto | l'unico rimasto che cambia la geometria |

Totale misurato: **~+3,8%** di velocità, contro la stima di §6 di +5/8% per
l'insieme §1-§4. La stima era ottimistica ma dell'ordine giusto.

⚠️ **Due lezioni di misura, pagate sul campo:**

1. §3 richiedeva prima di ribasare i punteggi di matto da `INT32_MAX` a
   `MATE_VALUE = 32000` (`ccf7612`), perché il payload era pieno: i 16 bit per la
   static eval vengono dal campo score. Il piano dava per scontato che i
   punteggi stessero già in int16 — **era falso in questo engine.**
2. §1+§4 e §3 lasciano l'albero **bit-identico**, quindi non servono SPRT: il
   tempo è una misura pulita. Ma il beneficio multi-thread dell'XOR — la sua
   giustificazione teorica principale — **non è stato dimostrabile**: a 4 thread
   la dispersione run-to-run è ~2x e sommerge l'effetto. Resta non verificato.

⚠️ Ribasare i matti ha anche rivelato che la **mate-distance pruning era codice
morto** (bound a ~2,1 miliardi dai punteggi reali): riattivata in `96337ef`,
−74% di nodi sulla suite di matti.

---

⚠️ `tt/tt.hpp` è marcato in `CLAUDE.md` fra i file da non toccare senza capirli:
concorrenza seqlock, sicurezza Lazy SMP, contratto di formato. Ogni intervento
qui richiede bench6, SPRT **e** una prova a più thread.

---

## 0. Perché il profilo a istruzioni mentiva

Callgrind (conteggio istruzioni) e `perf` (conteggio cicli) danno due classifiche
diverse, e solo la seconda descrive il tempo reale:

| funzione | istruzioni | **cicli** |
|---|---|---|
| `searchPosition` | 23,6% | 22,7% |
| **`TT::probeEntry`** | **1,6%** | **13,7%** |
| `staticExchangeEvaluation` | 8,6% | 12,0% |
| `NNUE::evaluate` | 7,6% | 7,5% |
| `Accumulator::update<true>` | 7,0% | 7,3% |
| `sortLegalMoves` | 12,0% | 7,0% |
| `Accumulator::update<false>` | 6,9% | 6,3% |
| undo mosse quiescenti | 8,2% | 5,9% |
| `TT::store` | — | 3,5% |

`probeEntry` passa da 1,6% a 13,6%: **un fattore 8,5**. Non esegue quasi nulla,
aspetta la memoria. Con `store` la TT è il **17,2% dei cicli**, seconda voce dopo
la ricorsione stessa.

Lezione di metodo: per una struttura dati legata alla memoria, un profilo a
istruzioni è attivamente fuorviante. `perf` richiede
`sudo sysctl -w kernel.perf_event_paranoid=1`.

---

## 1. Ogni probe tocca DUE linee di cache — il difetto principale

```cpp
allocBytesFor(buckets) = buckets * 64   // array Entry
                       + buckets * 4;   // array BucketSeq  <-- regione separata
```

Il bucket occupa 64 byte esatti (4 entry × 16 B), quindi il seqlock **non ci sta
dentro** e vive in un array separato, megabyte più in là. `findEntrySnapshot`
legge il bucket **e** `bucketSeq[bucketIndex]`: due miss indipendenti e due voci
di TLB per nodo.

`prefetch()` è corretto e li precarica entrambi, ma questo raddoppia le richieste
in volo e la pressione sul TLB — la risorsa già satura senza huge page.

**Rimedio: XOR-lockless.** Memorizzare `key ^ payload` al posto della chiave e
verificare in lettura. Una lettura strappata produce una chiave che non combacia
→ trattata come miss, che è sempre sicuro. È lo schema di Stockfish e della
maggior parte dei motori moderni.

Elimina in un colpo: la seconda linea di cache, la seconda voce di TLB, il
secondo prefetch, il ciclo di retry, il CAS di `store` (§4) e il 6,25% di memoria
dell'array seq.

⚠️ Controintuitivo ma importante: risulterebbe **meno** codice concorrente di
adesso, non di più. Il seqlock è la parte delicata di questo file; lo schema XOR
non ha lock affatto.

---

## 2. Metà di ogni entry è chiave

```cpp
struct Entry { uint64_t key; uint64_t payload; };   // 8 + 8
```

Otto byte su sedici servono solo a verificare l'identità, mentre l'indice del
bucket fornisce già 20 bit di discriminazione (a 1M bucket). Conservare **16 bit**
di chiave porta l'entry a 8 byte: **8 entry per linea di cache** invece di 4,
oppure metà footprint a parità di entry — riduzione diretta di miss e di TLB.

Rischio governato: una collisione produce una mossa di hash illegale o un
punteggio sbagliato, e **entrambi i casi sono già gestiti** (la mossa di hash
viene validata contro le mosse legali; i bound sono comunque euristici).

---

## 3. Manca il campo static-eval, ed era pianificato

Il payload è pieno **esattamente**:

```
score(32) | bestMove(16) | packed depth+flag+age(16)  = 64 bit
```

Ma il punto #3 di `HOTPATH_IMPROVEMENTS.md` prometteva: *"qui si aggiungerà il
campo static-eval nel payload quando arriverà NNUE"*. Mai fatto. E
`NNUE::evaluate` è il **7,5% dei cicli**: ogni nodo con hit in TT che ricalcola
la valutazione statica per RFP o NMP rifà un lavoro già svolto.

Lo spazio esiste: **il punteggio non ha bisogno di 32 bit.** I valori stanno
comodamente in `int16_t`, mate compresi (±32000). Restringerlo libera 16 bit per
l'eval statica senza allargare l'entry di un byte.

Da fare **insieme** a §2, perché toccano lo stesso layout: una sola migrazione di
formato, un solo ciclo di validazione.

---

## 4. `store()` paga un CAS su una linea diversa

`lockBucket` esegue `compare_exchange_weak` — una `lock cmpxchg` che serializza,
~20 cicli anche senza contesa, **sulla linea del seqlock**, diversa da quella dei
dati. `TT::store` è il 3,5% dei cicli e buona parte è questo.

Con lo schema XOR di §1 lo store diventa due scritture semplici.

---

## 5. Cosa NON è un problema (non perderci tempo)

- **`atomic_ref` con `memory_order_relaxed`**: su x86 compilano in load/store
  normali. Costo zero.
- **Ciclo di retry `MAX_RETRIES = 8`**: a thread singolo non ritenta mai.
- **Allineamento**: `align_val_t(64)` sull'heap, mmap allineata di suo. I bucket
  sono su linea, nessun accesso a cavallo. Corretto.
- **Indicizzazione** con i bit bassi della chiave Zobrist: ben distribuita.
- **Politica di rimpiazzo** `(ageDiff << 8) - (depth << 2)` con preferenza per gli
  slot vuoti: sensata.
- **Unificazione dei probe**: già fatta (era il punto #5 dei candidati alla
  rimozione). `probeEntry` è l'unica lettura calda; `probeMove` /
  `probeDecodedMove` sopravvivono solo per ponder e ricostruzione della PV,
  entrambi fuori dall'albero.

---

## 6. Quanto vale — con scetticismo

⚠️ **Non sommare ingenuamente i tre interventi.** La linea del *bucket* resta
comunque un miss: togliere il seqlock elimina il secondo miss, non il primo.

Stima realistica dell'insieme: **+5/8% di NPS**, cioè **+4/6 Elo** con la regola
del progetto (+10% velocità ≈ +7/10 Elo).

Per confronto, il batch di finali del 2026-08-04 ha dato **+27,9 Elo in una
giornata**. Finché esistono buchi nei *dati* di quella taglia, la priorità sta
lì. Questo piano è per quando quel filone sarà esaurito.

---

## 7. La leva a costo zero: huge page

Attacca lo stesso 13,7% dal lato TLB, **senza toccare una riga di codice**.

Stato misurato il 2026-08-04 su questa macchina:

- `HugePages_Total: 0` → `MAP_HUGETLB` fallisce, il codice ripiega su THP
- `AnonHugePages: 0 kB` sulla regione della TT → **il ripiego non produce nulla**
- Un test minimale in C con mmap allineata a 2 MiB ottiene anch'esso 0 → **non è
  colpa del motore**: il sistema, dopo giorni di uptime e decine di GB di churn,
  non trova più blocchi fisici contigui da 2 MiB

Con hash da 64 MiB su pagine da 4 KiB servono 16.384 voci di TLB per coprire la
tabella, contro le poche migliaia disponibili: **ogni probe paga un page walk
oltre al cache miss**.

Rimedio, che è la *prima* scelta del codice e oggi fallisce:

```sh
sudo sysctl -w vm.nr_hugepages=64      # 128 MiB, copre l'hash da 64
```

Poi rilanciare e verificare `AnonHugePages` o `HugePages_Free`. Atteso **+1/3% di
NPS**. Da misurare a macchina libera, prima di qualunque lavoro di §1-§4.

---

## 8. Ordine consigliato

1. **Huge page** — zero codice, zero rischio, misurabile in un'ora
2. **§2 + §3 insieme** — chiave a 16 bit ed eval statica nel payload: una sola
   migrazione di formato
3. **§1 + §4** — XOR-lockless: toglie seqlock, CAS e seconda linea di cache

Ognuno dei tre gruppi va validato con bench6 (il conteggio nodi **cambierà**: la
TT influenza l'albero), SPRT sotto controllo di tempo, e una prova a 4 thread per
verificare che la sicurezza Lazy SMP regga.
