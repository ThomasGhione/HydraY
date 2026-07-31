#!/usr/bin/env bash
#
# run_nodes_ab.sh — A/B test: datagen a 8k vs 12k nodi/mossa, a PARITÀ DI TEMPO.
#
# Domanda a cui risponde: un'ora di datagen rende di più a 8k (più posizioni)
# o a 12k (meno posizioni, etichette migliori)? Le due opzioni si confrontano
# addestrando due reti shakedown identiche e facendole giocare in SPRT.
#
# PIPELINE (fasi = sottocomandi):
#   1. ./tuning/run_nodes_ab.sh datagen [ore_per_braccio]   (default 6)
#        Genera i due dataset IN SEQUENZA sulla stessa macchina (stesso tempo,
#        stesso carico). Etichette dalla rete embedded nel ./chess corrente
#        (= v2), syzygy adjudication attiva. Ripetibile: i run si accumulano
#        (datagen appende), basta dare le stesse ore a entrambi i bracci.
#        Multi-macchina: lanciare la stessa fase su ogni macchina (i file
#        hanno suffisso hostname), poi copiare tutto in nnue/data e fare prep.
#   2. ./tuning/run_nodes_ab.sh prep
#        Tronca a multipli di 32 B, scarta i record azzerati (power-loss),
#        shuffle globale, statistiche W/D/L, comprime in .zst pronti per Drive.
#   3. Training su Colab (manuale, 2 run shakedown da 10 superbatch):
#        colab_ob.ipynb (branch output-buckets — arch corrente) con dataset
#        ab8k_shuffled.bin.zst → net_id ab8k; poi ab12k_shuffled → ab12k.
#        Le reti devono pesare 803.904 B (789.568 = arch vecchia, run errato).
#        Scaricare i due quantised.bin.
#   4. ./tuning/run_nodes_ab.sh sprt <ab8k.bin> <ab12k.bin>
#        Stesso binario da entrambi i lati, solo EvalFile diverso.
#        H1 accettata → 12k etichetta meglio: v3 a 12k.
#        H0 accettata → il vantaggio non ripaga il -25/40%% di velocità: v3 a 8k.
#
#   ./tuning/run_nodes_ab.sh status   # posizioni raccolte finora per braccio
#
# Knob via env: THREADS (default 3, come datagen), ARMS (default "8000 12000").
# ---------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
data_dir="${repo_root}/nnue/data"
chess_bin="${repo_root}/chess"
host="$(hostname -s)"

THREADS="${THREADS:-3}"
ARMS="${ARMS:-8000 12000}"

mkdir -p -- "${data_dir}"   # /nnue/data/ è gitignored: sui clone freschi non esiste

arm_tag() { # 8000 -> ab8k
    echo "ab$(( $1 / 1000 ))k"
}

# --- datagen [ore] -----------------------------------------------------------
cmd_datagen() {
    local hours="${1:-6}"
    [[ -x "${chess_bin}" ]] || { echo "errore: ${chess_bin} mancante — make prod" >&2; exit 1; }

    echo "== A/B datagen: $(echo ${ARMS} | wc -w) bracci × ${hours}h, ${THREADS} thread, host ${host} =="
    echo "   (macchina il più possibile scarica: i bracci girano in sequenza per essere comparabili)"
    for nodes in ${ARMS}; do
        local tag prefix
        tag="$(arm_tag "${nodes}")"
        prefix="${data_dir}/${tag}_${host}"
        echo
        echo "-- braccio ${tag} (${nodes} nodi/mossa) → ${prefix}.t*.bin, ${hours}h --"
        # datagen gestisce SIGTERM in modo pulito (flush e stop); -k è la rete
        # di sicurezza. timeout esce 124 a scadenza: non è un errore.
        timeout --foreground --signal=TERM -k 120 "${hours}h" \
            "${chess_bin}" datagen "${prefix}" "${THREADS}" "${nodes}" \
            || [[ $? -eq 124 ]]
    done
    echo
    cmd_status
}

# --- status -------------------------------------------------------------------
cmd_status() {
    echo "== posizioni raccolte (tutti gli host presenti in nnue/data) =="
    for nodes in ${ARMS}; do
        local tag total f sz
        tag="$(arm_tag "${nodes}")"
        total=0
        for f in "${data_dir}/${tag}"_*.t*.bin; do
            [[ -e "${f}" ]] || continue
            sz=$(stat -c%s "${f}")
            total=$(( total + sz / 32 ))
        done
        printf "  %-6s %'d posizioni\n" "${tag}" "${total}"
    done
}

# --- prep ----------------------------------------------------------------------
cmd_prep() {
    for nodes in ${ARMS}; do
        local tag out
        tag="$(arm_tag "${nodes}")"
        out="${data_dir}/${tag}_shuffled.bin"
        echo "== prep ${tag} → ${out} =="
        python3 - "${out}" "${data_dir}/${tag}"_*.t*.bin <<'PYEOF'
import random, sys, struct

out_path, in_paths = sys.argv[1], sys.argv[2:]
REC = 32
records = []
for p in in_paths:
    with open(p, "rb") as f:
        blob = f.read()
    usable = len(blob) - (len(blob) % REC)   # il writer appende: coda parziale via
    if usable != len(blob):
        print(f"  {p}: troncati {len(blob) - usable} B di coda parziale")
    records.append(blob[:usable])

blob = b"".join(records)
n = len(blob) // REC
mv = memoryview(blob)

# Scarta record azzerati (power-loss a metà scrittura: occ == 0 è impossibile
# in una posizione vera) e conta i risultati (result @ offset 26: 0/1/2).
keep, wdl = [], [0, 0, 0]
for i in range(n):
    off = i * REC
    if mv[off:off + 8] == b"\x00" * 8:
        continue
    keep.append(i)
    wdl[mv[off + 26]] += 1

dropped = n - len(keep)
random.seed(0xAB)                            # shuffle riproducibile
random.shuffle(keep)

with open(out_path, "wb") as f:
    for i in keep:
        f.write(mv[i * REC:(i + 1) * REC])

tot = len(keep)
print(f"  {tot:,} posizioni ({dropped} azzerate scartate)  "
      f"W/D/L stm: {wdl[2]*100//tot}/{wdl[1]*100//tot}/{wdl[0]*100//tot}")
PYEOF
        # spot-check leggibilità col reader dell'engine
        "${chess_bin}" datagen-dump "${out}" 3 >/dev/null \
            && echo "  datagen-dump: OK"
        zstd -T0 -f --rm -q "${out}" -o "${out}.zst"
        echo "  pronto per Drive: ${out}.zst ($(du -h "${out}.zst" | cut -f1))"
    done
    echo
    echo "Prossimo passo: 2 run shakedown su Colab (colab_ob.ipynb, 10 superbatch,"
    echo "net_id ab8k / ab12k), poi: ./tuning/run_nodes_ab.sh sprt <ab8k.bin> <ab12k.bin>"
}

# --- sprt <net8k> <net12k> ------------------------------------------------------
cmd_sprt() {
    [[ $# -eq 2 ]] || { echo "uso: run_nodes_ab.sh sprt <ab8k.bin> <ab12k.bin>" >&2; exit 1; }
    local net8k net12k
    net8k="$(realpath "$1")"
    net12k="$(realpath "$2")"
    [[ -f "${net8k}" && -f "${net12k}" ]] || { echo "errore: rete mancante" >&2; exit 1; }
    [[ -x "${chess_bin}" ]] || { echo "errore: ${chess_bin} mancante — make prod" >&2; exit 1; }

    # Stesso binario da entrambi i lati: conta solo la rete.
    cp -- "${chess_bin}" "${script_dir}/chess_baseline"
    echo "== SPRT: NEW = 12k (${net12k})  vs  BASE = 8k (${net8k}) =="
    echo "   H1 → v3 a 12k nodi/mossa; H0 → restare a 8k."
    NEW_OPTS="EvalFile=${net12k}" BASE_OPTS="EvalFile=${net8k}" \
        "${script_dir}/run_sprt.sh"
}

# --- dispatch -------------------------------------------------------------------
case "${1:-}" in
    datagen) shift; cmd_datagen "$@" ;;
    status)  cmd_status ;;
    prep)    cmd_prep ;;
    sprt)    shift; cmd_sprt "$@" ;;
    *)       sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
