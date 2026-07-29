# HALFKA_DECISION_PLAN.md — decidere HalfKA, e cosa fare del datagen intanto

Piano operativo aperto il **2026-07-28**. Copre due binari che procedono in
parallelo e non si bloccano a vicenda:

- **Binario A — decisione HalfKA**: serve solo Colab, **zero posizioni nuove**.
- **Binario B — qualità del datagen**: serve solo la macchina locale.

Documenti di riferimento: `HALFKA_PLAN.md` (architettura e fasi F1-F6),
`DATAGEN_QUALITY_PLAN.md` (§3 protocollo nodi/mossa, §5 buco bucket-0),
`NNUE_PLAN.md` (roadmap generale).

---

## 0. Premessa: perché il primo confronto HalfKA non conta

Il 466M-vs-HalfKA che ha dato **−40/−70 Elo** non misurava l'architettura:

| | superbatch | parametri l0 |
|---|---:|---:|
| HalfKA | 10 (shakedown) | ×4 |
| v3 ob | 40 (full) | ×1 |

Un quarto dei passi di gradiente per una rete quattro volte più grande, cioè
**~1/16 di addestramento per parametro**. Perdere è l'esito atteso: il
risultato non dice nulla su HalfKA.

Cause alternative già **escluse** (verificate il 2026-07-28):

- **Factoriser presente e attivo** — `l0f` condiviso, `repeat(INPUT_BUCKETS)`,
  sommato a `l0`, fuso nei pesi al salvataggio, clipping ±0,99 su entrambi.
- **Correction history in parità** — presente sia su `dev` sia su `halfka`.
- **Search in parità** — i 31 commit di `dev` assenti da `halfka` sono
  datagen/test/driver/refactor: nessuno cambia comportamento di ricerca.

Causa **non ancora esclusa**: F5 non fatta, quindi HalfKA paga un handicap NPS
stimato dal piano in **−5/−15%** (~5-7 Elo a tempo fisso) che si somma a tutto.

> ⚠️ Correggere in `HALFKA_PLAN.md` la riga F4 (*"Attesa: già positivo o quasi;
> se disastro → bug, fermarsi"*): l'attesa è irrealistica perché confronta uno
> shakedown a 10 SB con la rete ob addestrata piena. È la formulazione che ha
> portato a leggere come "HalfKA non funziona" un risultato non informativo.

---

## Binario A — decisione HalfKA

### A0. Unire i dataset di tutte le macchine — ✅ **FATTO** (2026-07-29)

Risultato: `nnue/data/hydray_v4_2019M_shuffled.bin`, **2.018.993.214 posizioni**
(64,6 GB), merge + shuffle globale in 6m17s con 8,2 GB di RAM di picco.

| sorgente | posizioni |
|---|---|
| `portatile2_v4` (16 thread) | 1.384.275.161 |
| `portatile2_v4b` (16 thread) | 139.321.834 |
| `andrea1` | 234.337.170 |
| `andrea2` | 261.059.049 |
| **totale** | **2.018.993.214** |

```sh
cargo run -r --bin datastats   -- <shard...>   # composizione, duplicati
cargo run -r --bin datashuffle -- <out.bin> <shard...>
```

**Omogeneità dell'etichettatore: verificata.** Le distribuzioni dei quattro
sorgenti coincidono a un decimale su ogni bucket, quindi provengono tutte dalla
rete v3:

| sorgente | W/D/L stm | \|score\| 0-49 / 200-399 / 800-3000 |
|---|---|---|
| `portatile2_v4` | 36,0 / 27,7 / 36,3 | 11,2% 20,0% 25,5% |
| `andrea1` | 36,1 / 27,6 / 36,3 | 11,2% 20,0% 25,5% |
| `andrea2` | 36,1 / 27,6 / 36,3 | 11,2% 20,0% 25,5% |
| `portatile2_v4b` | 36,2 / 27,4 / 36,4 | 11,3% 20,0% 25,6% |

**`portatile2_v3` (122M) ESCLUSO**: generazione precedente, etichettata da un
net diverso, e si misura — patte al **25,5%** contro il 27,6% dei v4, con lo
shift corrispondente su tutto l'istogramma. Sono i target in conflitto che
questo controllo doveva intercettare.

Integrità: 0 record azzerati, 0 code parziali, conteggio conservato esattamente
fra pass 1 e pass 2. Il dup-rate scende da 1,0–1,5% nei sorgenti a **0,13%** su
una fetta dell'output: i duplicati erano interni a partita/thread e ora sono
sparsi, che è la proprietà utile per l'SGD.

⚠️ **Non cancellare i `.bin` grezzi**: lo shuffle è globale e non appendibile,
quindi l'arrivo dei 700M della terza macchina impone un rimerge dai sorgenti.

### A1. Confronto a budget pari — ✅ **FATTO** (2026-07-29): HalfKA vince

**Esito: +68,6 Elo per HalfKA**, 453/760 (59,6%), IC 95% ≈ ±22 — il limite
inferiore resta sopra +45. SPRT fermato a 760 partite per scelta: il gate era
superato con ampio margine e i core servivano per A2.

⚠️ **Non usare 68,6 come stima dell'effetto reale.** È gonfiata dall'arresto
opzionale ed è il confronto fra due reti **sotto-addestrate**; non predice il
guadagno a budget pieno. Ciò che A1 stabilisce è **l'ordinamento**, ed è
inequivocabile.

Questo **ribalta lo shakedown** (−40/−70 Elo) e conferma la diagnosi: il difetto
era il regime a 2,1 epoche su 466M con ×4 parametri, non l'architettura.

Segnali concordanti a budget identico:

| | 768 | HalfKA |
|---|---|---|
| running loss (10 SB) | 0,014430 | **0,014238** |
| sanity startpos | 42 cp | 41 cp |
| mirror (wN e4 / bN e5) | 805 = 805 | 765 = 765 |

**Parità di search ottenuta, non stimata.** HalfKA tocca solo `board/` e `nnue/`
— zero righe in `engine/search/` — e `dev` aveva modificato quei file per 15
righe. Il merge di `dev` in `halfka` (branch `halfka-a1`) è passato senza
conflitti e `git diff dev halfka-a1 -- engine/ uci/ tt/` è **vuoto**. L'unica
variabile è l'architettura NNUE.

Nota di build: ogni binario deve avere il **proprio** net embedded.
`embeddedNetwork()` fa un reinterpret senza controllo di dimensione, quindi il
binario HalfKA col net 768 da 803 KB leggerebbe 3,1 MB fuori dai limiti.
`nnue-selftest` sul binario HalfKA: **41.577 posizioni, incremental ≡ scratch**.

### A1-bis. Il bucket 0 è vuoto in entrambe le reti

`KQvK` (3 pezzi) vale **−47 cp** su HalfKA e **0 cp** sulla 768: donna contro re
nudo valutata zero. Stesso quadro su `active kings endgame` (4 pezzi) e
`KRPvKR` (5 pezzi). È il bucket di output 0 affamato dall'adjudication Syzygy,
che chiude le partite proprio a ≤5 pezzi. Difetto **dei dati**, identico per le
due architetture: non falsa A1, ma è Elo perso adesso in entrambe le direzioni.
Lo risolve **B1** (§5 di `DATAGEN_QUALITY_PLAN.md`).

### A1-storico. Impostazione del confronto

Addestrare l'architettura **attuale** (768→512 + 8 ob) e **HalfKA** con lo
**stesso numero di superbatch** sullo **stesso dataset unito ~2B**.

```sh
# stesso data.bin, stesso budget, per entrambe le architetture
cargo run -r --bin trainer --features cuda -- /content/data.bin 10 hydray-768-sb10
cargo run -r --bin trainer --features cuda -- /content/data.bin 10 hydray-halfka-sb10
```

**Perché non sui 466M.** Un superbatch è ~100M campioni
(`batch_size 16_384 × batches_per_superbatch 6_104`), quindi il dataset
determina le **epoche**:

| dataset | 10 SB | 40 SB |
|---|---|---|
| 466M | 2,1 epoche (dati ripetuti) | 8,6 epoche |
| **1B (prefisso, A1)** | **1,0 epoca** | 4 epoche |
| 2,019B (unito, A0) | 0,5 epoche | 2 epoche |

Lo shakedown originale ha fatto 2,1 epoche su 466M con una rete a ×4 parametri:
condizioni da overfitting. A 1,0 epoca ogni passo vede dati nuovi, che è il
regime in cui la capacità extra di HalfKA può ripagare. Rifare A1 sui 466M
sottostimerebbe HalfKA per costruzione.

**Perché il prefisso da 1B e non i 2,019B interi.** Il runtime Colab ha ~78 GB
di disco locale e il notebook decomprime in `/content/data.bin`: i 64,6 GB del
dataset unito lascerebbero ~13 GB per checkpoint e sistema, troppo stretto.
Il file unito è shuffolato **globalmente**, quindi qualsiasi prefisso è un
campione uniforme — `head -c 32000000000` non introduce bias. Per A1, che è un
confronto ad armi pari, 1,0 epoca basta e avanza.

Dataset A1: `nnue/data/hydray_v4_1000M_shuffled.bin.zst` (prefisso di
`hydray_v4_2019M_shuffled.bin`, 1.000.000.000 posizioni).

Confronto: costruire i due binari (HalfKA dal branch `halfka`, 768 da `dev`),
congelare il 768 come baseline e lanciare l'SPRT.

```sh
git switch dev && make prod && ./tuning/run_sprt.sh --snapshot
git switch halfka && make prod
./tuning/run_sprt.sh          # ELO0=0 ELO1=5
```

**Gate A1**
- HalfKA **≥** 768 a parità di budget → segnale incoraggiante, si va ad A2/A3.
- HalfKA perde di **poco** (entro ~15 Elo) → plausibile che sia l'handicap NPS:
  fare A2 **prima** di concludere.
- HalfKA perde **nettamente** (>30 Elo) a budget pari → l'architettura non
  rende su questo motore; passare direttamente a **A5**.

### A2. F5 — FinnyTable e misura NPS — ✅ **FATTO** (2026-07-29)

Le Finny table erano già implementate sul branch (`c7abccb`, F5). Restava la
misura. bench6 a profondità 12, 3 giri interleaved a macchina quieta:

| | nodi | tempo | NPS |
|---|---|---|---|
| 768 | 3.826.701 | 1,18 s | 3,23 M |
| HalfKA | 3.650.818 | 1,18 s | 3,10 M |

**Penalità NPS: solo −4,2%**, e HalfKA cerca **−4,6% di nodi** per raggiungere la
stessa profondità: i due effetti si annullano e il **time-to-depth è identico**.

Conseguenza per A1: i +68,6 Elo **non hanno penalità di velocità da scontare**.
La preoccupazione che motivava A2 — "ogni SPRT confonde architettura e
velocità" — non si applica, perché la velocità è alla pari.

### A3. Confronto definitivo a budget pieno (= F6)

Solo se A1 non ha già chiuso la questione. Prima **rebasare `halfka` su `dev`**
così l'unica differenza è la rete (i 31 commit sono neutri, ma il rebase
elimina la domanda).

```sh
cargo run -r --bin trainer --features cuda -- <data.bin> 40 hydray-halfka-full
cargo run -r --bin trainer --features cuda -- <data.bin> 40 hydray-768-full
```

Validazione prima dello swap: `sanity.rs` → swap rete → `nnue-selftest` →
nuovo baseline bench6 → SPRT → gauntlet.

**Gate A3**
- HalfKA vince → procedere verso la release, e la domanda sul volume dati
  diventa "quanti bucket posso permettermi" (vedi A4).
- HalfKA perde → **A5**.

### A4. Curva di apprendimento — solo se serve

Da fare **solo** se A3 dà HalfKA perdente o marginale. Addestrare HalfKA su
100M / 233M / 466M a budget fisso e generoso, guardando la validation loss:

- pendenza ancora ripida a 466M → si è sul tratto ripido, più dati aiutano
  davvero e il target volumetrico ha una base;
- curva piatta → il collo di bottiglia non sono i dati.

### A4-bis. Mappa a 8 bucket — ora è un candidato vicino, non "dopo"

`HALFKA_PLAN.md:41` ha scelto **4 bucket e non 8-13 perché il dataset era
300-400M** («mappe più fitte sono data-hungry e si possono A/B-are DOPO»). Con
**~2B quel vincolo è caduto**: sono 5-6× i dati che motivavano la scelta, ed è
nelle mappe più fitte che HalfKA produce il grosso del suo guadagno. La mappa è
un iperparametro puro — il codice engine-side non cambia.

**Disciplina però**: non muovere due variabili insieme. Prima HalfKA-4 contro
768 a budget pari su 2B (A1/A3); *poi*, se il segnale è positivo, A/B fra 4 e 8
bucket. Se A1 dà HalfKA-4 marginale, vale comunque la pena provare 8 prima di
scartare l'architettura: il vincolo che la teneva a 4 non esiste più.

### A5. Se HalfKA è davvero peggiore

Nessun dato è sprecato: `HALFKA_PLAN.md:97` — *"bulletformat invariato
(ksq/opp_ksq già presenti): il dataset va bene così com'è, nessuna dipendenza
dall'architettura"*. Precedente diretto: **v3-ob ha fatto +22,7 SPRT sugli
stessi 366M dati v2**, cambiando solo l'architettura.

Il dataset si riversa sull'architettura attuale, dove la leva naturale per
spendere più dati è aumentare l'hidden layer (768→1024) — al costo di NPS, da
misurare.

---

## Binario B — qualità del datagen (in parallelo, nessuna dipendenza da A)

### B1. Fix del buco bucket-0 — ✅ **FATTO** (2026-07-29, `4d2a157` + `a8b4af6`)

Misurato prima di intervenire: su 300k record del merge da 2B il bucket 0 è allo
**0,00%** — vuoto, non scarso. Da qui `KQvK = 0 cp` sulla 768 addestrata proprio
su quei dati.

**Il seeding da solo non bastava**, ed è il motivo per cui il difetto è
sopravvissuto al ciclo v4. Tre filtri lo proteggevano e andavano tolti insieme:

1. l'adjudication Syzygy chiude la partita appena si entra in range TB;
2. un finale vinto vale **punteggio di matto**, che `MAX_RECORD_SCORE_CP` scarta
   — `KQvK` non sarebbe entrata *nemmeno* con le tablebase spente;
3. `MIN_RECORD_PLY = 16` taglia i primi 16 ply, e un finale seminato è deciso
   prima.

Le partite seminate ora registrano da ply 0, saltano adjudication e filtro di
bilanciamento, e **clampano** invece di scartare.

Due correzioni venute dalla misura, non dal design:
- colori indipendenti per pezzo → materiale bilanciato → **70% patte, 96% dei
  record sotto 100 cp**: insegnerebbe "sotto i 5 pezzi è tutto pari". Ora il 65%
  dei semi dà tutti i pezzi extra a un lato solo.
- le partite decise finiscono in pochi ply, le patte tirano fino alle 50 mosse:
  senza un tetto per partita le patte inondavano i record comunque.

Resa misurata (tablebase attive): run misto 1-su-8 → bucket 0 all'**1,54%**;
**run dedicato** (`CHESS_DATAGEN_EG_EVERY=1`) → **88,8%**, cioè ~20× per ora di
macchina. Il run misto serve al datagen lungo, quello dedicato a riempire il
bucket in una finestra corta.

⚠️ **I 2B già generati non sono riparati**: il fix vale solo per i dati nuovi.
Il lotto `portatile2_eg` esiste per questo. In un mix da ~2,7B pesa ~0,2%:
abbondante per i 1024 pesi di output di quel bucket, ma il salto vero è da
*zero* a *qualcosa*.

**Verifica finale, dopo A3**: `KQvK` deve smettere di valere 0 cp.

### B2. A/B nodi/mossa (§3)

Oggi 8000 nodi. Il piano stima che 10-12k dia etichette migliori al 25-40% di
costo in velocità. La domanda vera non è quanti miliardi ma **se 2B a 12k valga
più di 5B a 8k**. Protocollo già definito nel §3: due shakedown identiche su
dataset prodotti coi due valori, poi SPRT.

Da fare su **una sola macchina** per non compromettere le altre.

### B3. Target volumetrico — decisione sospesa

Il `5B` nel codice è **solo la riga di ETA** (`8e0e3ef`,
`TARGET_POSITIONS ... // v4 target (ETA line only)`): niente si ferma a quel
valore, niente si rompe prima.

I ritorni sulla quantità di dati sono ~logaritmici: ogni raddoppio compra un
incremento circa costante, quindi il quarto e quinto miliardo sono i più
deboli. L'unico punto sperimentale pulito è v2→v3 (366M→464M, +11,5 Elo
stimati), che estrapolato darebbe ~34 Elo/raddoppio — **ma è un punto solo,
possibilmente confuso con altri cambiamenti, e vale per l'architettura a 768,
non per HalfKA**.

**Con ~2B in mano l'estrapolazione non serve più: si misura.** Addestrare a
budget fisso su 466M / 1B / 2B dà tre punti della curva reale, sull'architettura
che interessa. Tre run brevi rispondono meglio di tre settimane di generazione,
e dicono anche *dove* la curva si appiattisce — cioè se il quarto e quinto
miliardo valgono il tempo macchina.

**Decidere dopo A1/A3 e B1/B2**, non prima: composizione ed etichette hanno più
leva del volume.

---

## Ordine consigliato

1. **B1** (fix bucket-0) — sblocca settimane di datagen utile, non aspetta nulla.
2. **A0** (unione + validazione dei ~2B) — prerequisito di tutto il binario A.
3. **A1** (confronto a budget pari su 2B) — risponde alla domanda che conta.
4. **A2** (Finny + NPS) — separa architettura da velocità.
5. **A3**, **A4-bis** (8 bucket) oppure **A5** secondo i gate.
6. **B2** (nodi/mossa) quando c'è una macchina libera da dedicarci.
7. **B3** solo alla fine, misurando la curva sui 2B invece di estrapolare.

## Regole di verifica (valgono per ogni passo)

- Ogni cambio all'accumulatore o agli hook della Board → **`./chess nnue-selftest`**.
- Ogni cambio a movegen/legalità/doMove → **`./chess perft suite`**.
- Refactor puri → **bench6 identico al nodo** (`script/engine_driver.sh bench6`).
- Forza → **solo SPRT sotto controllo di tempo**, mai a profondità fissa.
- Misure NPS → macchina quieta, run interleaved, datagen fermo.
- Il binario del branch `halfka` **non va mai deployato** al bot.
