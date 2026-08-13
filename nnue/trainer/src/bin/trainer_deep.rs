// HydraY NNUE v8 — rete con layer intermedi (progetto "layer intermedio").
//
// Architettura: (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 32 -> 1,
// con 8 output bucket su OGNI layer dopo il feature transformer.
// Adattato da bullet examples/progression/4_multi_layer.rs, che e' il seguito
// diretto di 3_input_buckets.rs da cui e' adattato trainer.rs.
//
// Sta in un binario SEPARATO da trainer.rs di proposito: la rete a un solo
// layer resta quella in produzione finche' questa non vince un SPRT, e le due
// devono poter essere addestrate senza toccarsi.
//
// COSA CAMBIA rispetto a trainer.rs, e cosa NO
//
// Cambia l'architettura, e con essa due cose che NON sono libere di restare
// come prima perche' fanno parte del pacchetto:
//
//   1. CReLU al posto di SCReLU sul feature transformer. La nonlinearita'
//      quadratica la fornisce il prodotto a coppie (pairwise): moltiplicare
//      fra loro le due meta' CReLU-ate da un termine di secondo grado E
//      dimezza la larghezza (1024 -> 512 per prospettiva). E' il motivo per
//      cui il layer costa poco: l'ingresso di l1 e' 1024, non 2048.
//   2. init_with_effective_input_size(32): inizializza i pesi con fan-in 32
//      (le feature attive per posizione) invece della dimensione nominale
//      dell'input. E' l'init corretto per ingressi sparsi e conta di piu' su
//      una rete profonda, dove un init sbagliato puo' impedire l'addestramento
//      invece di limitarsi a rallentarlo.
//
// NON cambia tutto il resto, deliberatamente: stessi 4 king bucket specchiati,
// stessi 8 output bucket, stesso WDL 0.3, stesso StepLR, stesso batch. Se
// questa rete perde, deve essere colpa dell'architettura e non di un
// iperparametro cambiato per distrazione.
//
// LAYOUT DEL FILE QUANTIZZATO (little-endian, nell'ordine di save_format):
//
//   l0w  [4*768][1024]  i16  QA=255   (factoriser gia' sommato)
//   l0b  [1024]         i16  QA=255
//   l1w  [8][16][1024]  i8   QB=64    (trasposto: ogni bucket contiguo)
//   l1b  [8][16]        f32
//   l2w  [8][32][16]    f32           (trasposto)
//   l2b  [8][32]        f32
//   l3w  [8][1][32]     f32           (trasposto)
//   l3b  [8]            f32
//
//   totale 6.443.552 B (la rete a un layer: 6.326.288 B)
//
// Solo l0 e l1w sono quantizzati: la coda 16->32->1 costa una manciata di
// operazioni per valutazione e in float evita ogni grattacapo di scala.
// sanity.rs va aggiornato PRIMA di fidarsi di un file prodotto da qui: e' il
// lettore di riferimento, e il C++ deve concordare con lui byte per byte.
//
// Uso identico a trainer.rs, incluse le tappe:
//   cargo run -r --bin trainer_deep --features cuda -- <data.bin> [superbatches]
//                                                      [net_id] [start_sb] [resume_ckpt]
//   STAGE_END=<n> limita la tappa corrente (vedi trainer.rs per il perche'
//   deve stare ATTACCATA a cargo e non in testa alla riga di shell).

use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

const HIDDEN_SIZE: usize = 1024;
// Larghezza dei due layer intermedi. L1 e' il costoso (1024 ingressi per
// bucket), L2 e' gratis. Partire da 16 e non da 32: il costo del forward
// scala lineare con L1_SIZE, e 16 dice gia' se il guadagno c'e'.
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;
const OUTPUT_BUCKETS: usize = 8;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

// Keep in sync with nnue/network.hpp KING_BUCKET_MAP e sanity.rs.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../data/hydray_16M_interleaved.bin".to_string());
    let superbatches: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let net_id = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "hydray-deep-shakedown".to_string());
    let start_superbatch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let resume_from = args.get(5).cloned();
    let stage_end: usize = std::env::var("STAGE_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(superbatches);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // i8 a QB=64: bullet ERRORE se un peso non ci sta (non tronca in
            // silenzio), quindi una scala sbagliata si manifesta al salvataggio
            // e non come una rete che gioca male senza motivo.
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(QB),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", HIDDEN_SIZE, OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, OUTPUT_BUCKETS);

            // Pairwise: CReLU sulle due meta' del feature transformer, poi
            // prodotto. Forma veloce di `.crelu().pairwise_mul()`. L'uscita e'
            // HIDDEN_SIZE/2 per prospettiva, quindi l'ingresso di l1 e'
            // HIDDEN_SIZE e non 2*HIDDEN_SIZE.
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, HIDDEN_SIZE / 2) * ft(stm_inputs, HIDDEN_SIZE / 2, HIDDEN_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, HIDDEN_SIZE / 2) * ft(ntm_inputs, HIDDEN_SIZE / 2, HIDDEN_SIZE);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    // Il peso SALVATO di l0 e' l0w + l0f: clippare entrambi a ±0.99 tiene la
    // somma dentro il ±1.98 sicuro per i16 a QA=255.
    let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter);
    trainer.optimiser.set_params_for_weight("l0f", stricter);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch,
            end_superbatch: stage_end,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.1,
            step: (superbatches / 2).max(1),
        },
        save_rate: superbatches.clamp(1, 10),
    };

    let settings = LocalSettings {
        threads: 2,
        test_set: None, // bullet al rev pinnato non la implementa: vedi trainer.rs
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data_path.as_str()]);

    if let Some(ckpt) = &resume_from {
        println!("resuming from {ckpt} at superbatch {start_superbatch}");
        trainer.load_from_checkpoint(ckpt);
    }

    trainer.run(&schedule, &settings, &data_loader);

    println!("=== sanity evals (stm-relative) ===");
    for (label, fen) in [
        ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
    ] {
        println!("{label}: {}", trainer.eval(fen));
    }
}

