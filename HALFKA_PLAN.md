# HalfKA + King Bucket — design (ciclo v4)

Obiettivo: input della rete **dipendenti dalla posizione del proprio re**, il
singolo salto architetturale più redditizio rimasto (+80-150 Elo tipici).
Architettura target: **(768×Nkb_hm → 512)x2 → 8ob** — king bucket mirrorati in
input, gli output bucket material-count già validati (+22,7) restano identici.

Principio ereditato dal ciclo ob: **prima il riferimento Rust, poi il C++,
cross-check byte-exact su rete sintetica PRIMA di addestrare.**

---

## 1. Feature set (deciso)

`ChessBucketsMirrored` di bullet (rev pinnata, `crates/bullet_lib/src/game/
inputs/chess_buckets.rs`) — semantica da replicare ESATTAMENTE engine-side:

```
per la prospettiva X (re proprio su ksq_X, già dal punto di vista di X, LERF):
  flip_X   = 7 se file(ksq_X) > 3, altrimenti 0     // mirroring orizzontale
  bucket_X = MAP[ksq_X]                              // mappa 64 (espansa da 32)
  feature  = 768 * bucket_X + (feat768 ^ flip_X)     // feat768 = isOpp*384+type*64+sq_X
```

Il re STA nei 768 (stile HalfKA, non HalfKP): una mossa di re che NON cambia
bucket né flip è un normale update incrementale della sua feature.

**Mappa iniziale: 4 bucket** (32 entry, file a-d per riga, rank 1 in alto):

```
0, 0, 1, 1,     // back rank: angolo/arrocco (g1,h1 → mirror → a1,b1) vs centro
2, 2, 2, 2,     // seconda traversa
3, 3, 3, 3,     // tutto il resto (re attivo / finale)
3, 3, 3, 3,
3, 3, 3, 3,
3, 3, 3, 3,
3, 3, 3, 3,
3, 3, 3, 3,
```

Perché 4 e non 8-13: con ~300-400M di posizioni v3 i parametri di l0 crescono
già ×4 (768×4×512); mappe più fitte sono data-hungry e si possono A/B-are DOPO
(la mappa è un iperparametro, il codice non cambia). Payload rete: ~3,2 MB
(embedded ok).

## 2. Trainer (bullet, riferimento: `examples/tests/test2.rs`)

- `.inputs(ChessBucketsMirrored::new(MAP))` + `.output_buckets(MaterialCount::<8>)`.
- **Factoriser**: `l0f` (768×H, `InitSettings::Zeroed`) ripetuto Nkb volte e
  sommato a l0 — la componente condivisa impara da TUTTI gli esempi, i bucket
  solo il residuo. Con i nostri volumi è quasi certamente un guadagno; al save
  bullet fonde il factoriser nei pesi quantizzati (verificare col sanity).
- Save format invariato nella sequenza (l0w | l0b | l1w^T | l1b); cambia solo
  la taglia di l0w: `Nkb*768` colonne.
- sanity.rs: aggiungere MAP + mirroring al calcolo feature (resta la spec del
  loader C++); estendere le FEN di riferimento con re in bucket diversi
  (arrocco corto/lungo, re centrale, re attivo in finale, PRIMA e DOPO
  l'arrocco).

## 3. Engine side (il grosso del lavoro)

### Accumulatore
Oggi: `Accumulator::update()` agnostico dal re, hook in add/removePieceFromBB,
undo exact-inverse senza snapshot. Con i king bucket ogni prospettiva dipende
da (bucket, flip) del PROPRIO re:

- mossa di pezzo ≠ re: update incrementale su entrambe le prospettive (indici
  calcolati col bucket/flip correnti di ciascuna) — come oggi.
- mossa del re di X con stesso (bucket, flip): update incrementale normale.
- mossa del re di X che cambia bucket o flip: **refresh completo della sola
  prospettiva X** (l'altra resta incrementale). Arrocco = re+torre, gestire
  l'ordine degli eventi.

### Undo
L'exact-inverse non basta più: l'undo di una mossa di re cross-bucket deve
ripristinare l'accumulatore calcolato col bucket vecchio. Due opzioni:
- (a) **refresh anche in undo** (semplice, costa un refresh in più; con Finny
  è ~cheap) ← partire da qui;
- (b) snapshot dell'accumulatore di X in MoveState solo per mosse cross-bucket
  (1 KB per lato, rare) — ottimizzazione se (a) pesa in NPS.

### FinnyTable (fase 2, dopo che il refresh-from-scratch funziona)
Cache per (prospettiva, bucket, flip): accumulatore + 12 bitboard dello stato
in cui fu calcolato. Refresh = diff pezzi vs stato cached (tipicamente pochi
update invece di ~30 add). Dimensioni: 2×4×2 slot × (512×2B + 96B) ≈ 18 KB
per Board → thread-safe by construction (ogni Board/helper ha la sua).

### Invarianti nuove da selftest
- incremental ≡ scratch dopo OGNI mossa di re (stesso bucket, cross-bucket,
  cross-flip, arrocco corto/lungo, entrambe le prospettive) e relativi undo;
- promozioni/en-passant vicino al re;
- il selftest attuale (random walk) copre già molto, ma aggiungere partite
  seedate con re erranti (endgame) per battere i bucket 2-3.

## 4. Dati e compatibilità

- bulletformat invariato (32 B, ksq/opp_ksq già presenti): **il dataset v3 in
  raccolta va bene così com'è** — nessuna dipendenza dall'architettura.
- datagen/selftest/uci: nessun cambio di interfaccia; cambia solo il payload
  della rete (nuovi static_assert in network.hpp, validazione taglia).

## 5. Fasi (ognuna con la sua validazione, stile ciclo ob)

- [ ] **F1 — trainer + sanity.rs** su branch `halfka`: arch nuova, MAP,
      factoriser; cargo check + sanity su rete sintetica generata a mano.
- [ ] **F2 — engine**: network.hpp (layout ×4), accumulatore king-aware con
      refresh-from-scratch su cambio bucket/flip (niente Finny ancora),
      refreshNnueAccumulator esteso.
- [ ] **F3 — validazione incrociata**: rete sintetica random → selftest
      (con i casi re nuovi) + cross-check C++ ≡ sanity.rs; bench6 gira.
- [ ] **F4 — shakedown**: 10 SB su ~50M del dataset v3 (Colab) → SPRT vs
      rete ob. Attesa: già positivo o quasi; se disastro → bug, fermarsi.
- [ ] **F5 — FinnyTable + NPS**: misurare prima/dopo (atteso −5-15% senza
      Finny, quasi-pari con); selftest di nuovo (la cache è il punto fragile).
- [ ] **F6 — run pieno**: 40 SB sul v3 completo → sanity → swap → selftest →
      bench6 nuovo baseline → SPRT vs ob → gauntlet → release.

## 6. Rischi noti

- **Undo/refresh** è il punto dove nascono i bug silenziosi: il selftest
  incremental≡scratch è l'unica difesa vera — estenderlo PRIMA di scrivere
  l'engine-side (TDD sul punto fragile).
- Factoriser: verificare che il quantised.bin fonda l0f+l0 (leggere il save
  path di bullet); se no, fonderlo nel sanity/loader.
- Lazy SMP: nessun problema atteso (accumulatore e Finny vivono nel Board,
  già per-thread), ma il bug latente 'stalls' va tenuto d'occhio nei run
  lunghi (vedi memoria project-latent-stalls-bug).
- La mappa a 4 bucket è un'ipotesi: A/B-arla solo DOPO che il pipeline è
  verde (stesso metodo del test 8k-vs-12k).
