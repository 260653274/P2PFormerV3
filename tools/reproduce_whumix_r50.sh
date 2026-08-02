#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${1:-${repo_dir}/checkpoints/p2pformer_corner_r50_whu_mix.pth}"
requested_split="${2:-all}"
dataset_root="${P2PFORMER_WHUMIX_ROOT:-${repo_dir}/../datasets/WHU-Mix}"
expected_sha256="07314e7358c0315343e7ba47dffc7a0e0bbc3a3addcb5ec22bd18c3064e1ce4c"
config="${repo_dir}/p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py"
output_root="${repo_dir}/work_dirs/reproduction"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Checkpoint SHA-256 mismatch: ${actual_sha256}" >&2
    exit 1
fi

run_split() {
    local split="$1"
    local annotation="$2"
    local image_dir="$3"
    local output_dir="${output_root}/r50_whu_mix_${split}"

    if [[ ! -f "${annotation}" || ! -d "${image_dir}" ]]; then
        echo "WHU-Mix ${split} data is incomplete under ${dataset_root}" >&2
        exit 1
    fi

    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${repo_dir}/tools/test.py" \
        "${config}" \
        "${checkpoint}" \
        --eval segm \
        --work-dir "${output_dir}" \
        --out "${output_dir}/results.pkl" \
        --cfg-options \
            "data.test.ann_file=${annotation}" \
            "data.test.img_prefix=${image_dir}/"
}

case "${requested_split}" in
    test1)
        run_split test1 "${dataset_root}/test1/test-1.json" \
            "${dataset_root}/test1/image"
        ;;
    test2)
        run_split test2 "${dataset_root}/test2/test-2.json" \
            "${dataset_root}/test2/image"
        ;;
    all)
        run_split test1 "${dataset_root}/test1/test-1.json" \
            "${dataset_root}/test1/image"
        run_split test2 "${dataset_root}/test2/test-2.json" \
            "${dataset_root}/test2/image"
        ;;
    *)
        echo "Usage: $0 [checkpoint] [test1|test2|all]" >&2
        exit 2
        ;;
esac
