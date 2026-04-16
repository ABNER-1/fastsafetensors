#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Unified test runner for fastsafetensors.
#
# Usage:
#   ./scripts/run_tests.sh --mock                  # Mock tests (no GPU/3FS needed)
#   ./scripts/run_tests.sh --3fs                   # 3FS environment tests
#   ./scripts/run_tests.sh --distributed           # Multi-process distributed tests
#   ./scripts/run_tests.sh --all                   # All of the above
#   ./scripts/run_tests.sh --mock --coverage       # With coverage collection
#   ./scripts/run_tests.sh --mock --framework paddle
#   ./scripts/run_tests.sh --distributed --world-size 2

set -uo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TESTS_DIR="${REPO_ROOT}/tests"

FRAMEWORK="${TEST_FASTSAFETENSORS_FRAMEWORK:-pytorch}"
WORLD_SIZE=4
COVERAGE=false
LOG_DIR="/tmp/fastsafetensors-test-logs"
MASTER_PORT=1234

# Scene toggles
RUN_MOCK=false
RUN_3FS=false
RUN_DISTRIBUTED=false

# Internal state
COVERAGE_IDX=0
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
FAIL_LIST=()

# ============================================================================
# Colors
# ============================================================================
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' RESET=''
fi

# ============================================================================
# Help
# ============================================================================
usage() {
    cat <<EOF
${BOLD}fastsafetensors test runner${RESET}

${BOLD}USAGE:${RESET}
    $(basename "$0") [OPTIONS]

${BOLD}SCENES:${RESET}
    --mock              Run mock tests (no GPU/3FS required)
                        Includes: test_fastsafetensors, test_parallel, test_vllm
    --3fs               Run 3FS environment tests
                        Includes: threefs/test_threefs, threefs/test_parallel_threefs, test_parallel
    --distributed       Run multi-process distributed tests (torchrun / paddle launch)
                        Includes: test_multi (1-node + N-node), test_parallel (torchrun)
    --all               Run all of the above

${BOLD}OPTIONS:${RESET}
    --framework NAME    Framework to test: pytorch (default) or paddle
    --world-size N      Number of nodes for distributed tests (default: 4)
    --coverage          Collect coverage and generate HTML report
    --log-dir DIR       Log directory (default: /tmp/fastsafetensors-test-logs)
    --help              Show this help message

${BOLD}EXAMPLES:${RESET}
    $(basename "$0") --mock
    $(basename "$0") --3fs --framework pytorch
    $(basename "$0") --distributed --world-size 2
    $(basename "$0") --all --coverage
    $(basename "$0") --mock --3fs

${BOLD}ENVIRONMENT VARIABLES:${RESET}
    TEST_FASTSAFETENSORS_FRAMEWORK   Override --framework (default: pytorch)
EOF
    exit 0
}

# ============================================================================
# Argument parsing
# ============================================================================
parse_args() {
    if [[ $# -eq 0 ]]; then
        usage
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mock)         RUN_MOCK=true ;;
            --3fs)          RUN_3FS=true ;;
            --distributed)  RUN_DISTRIBUTED=true ;;
            --all)
                RUN_MOCK=true
                RUN_3FS=true
                RUN_DISTRIBUTED=true
                ;;
            --framework)
                shift
                FRAMEWORK="${1:?--framework requires a value (pytorch|paddle)}"
                ;;
            --world-size)
                shift
                WORLD_SIZE="${1:?--world-size requires a number}"
                ;;
            --coverage)     COVERAGE=true ;;
            --log-dir)
                shift
                LOG_DIR="${1:?--log-dir requires a path}"
                ;;
            --help|-h)      usage ;;
            *)
                echo -e "${RED}Unknown option: $1${RESET}" >&2
                echo "Run '$(basename "$0") --help' for usage." >&2
                exit 1
                ;;
        esac
        shift
    done

    if ! $RUN_MOCK && ! $RUN_3FS && ! $RUN_DISTRIBUTED; then
        echo -e "${RED}Error: No test scene selected. Use --mock, --3fs, --distributed, or --all.${RESET}" >&2
        exit 1
    fi
}

# ============================================================================
# Environment detection
# ============================================================================
HAS_GPU=false
HAS_3FS=false
LIBDIR=""

check_environment() {
    echo -e "${BOLD}============================================================${RESET}"
    echo -e "${BOLD}  Environment Detection${RESET}"
    echo -e "${BOLD}============================================================${RESET}"

    # Framework
    echo -e "  Framework:    ${BLUE}${FRAMEWORK}${RESET}"

    # GPU
    if python3 -c "from fastsafetensors.common import is_gpu_found; exit(0 if is_gpu_found() else 1)" 2>/dev/null; then
        HAS_GPU=true
        echo -e "  GPU:          ${GREEN}available${RESET}"
    else
        echo -e "  GPU:          ${YELLOW}not available${RESET}"
    fi

    # 3FS
    if python3 -c "import fastsafetensor_3fs_reader" 2>/dev/null; then
        HAS_3FS=true
        echo -e "  3FS:          ${GREEN}available${RESET}"
    else
        echo -e "  3FS:          ${YELLOW}not available (mock mode)${RESET}"
    fi

    # LIBDIR for coverage
    LIBDIR=$(python3 -c "import os; os.chdir('/tmp'); import fastsafetensors; print(os.path.dirname(fastsafetensors.__file__))" 2>/dev/null || true)
    if [[ -z "$LIBDIR" ]]; then
        echo -e "  ${RED}Error: fastsafetensors not installed. Run 'pip install .' first.${RESET}" >&2
        exit 1
    fi
    echo -e "  LIBDIR:       ${LIBDIR}"

    # Coverage
    if $COVERAGE; then
        echo -e "  Coverage:     ${GREEN}enabled${RESET}"
    else
        echo -e "  Coverage:     disabled"
    fi

    # Scenes
    local scenes=""
    $RUN_MOCK && scenes+="mock "
    $RUN_3FS && scenes+="3fs "
    $RUN_DISTRIBUTED && scenes+="distributed "
    echo -e "  Scenes:       ${BLUE}${scenes}${RESET}"

    if $RUN_DISTRIBUTED; then
        echo -e "  World size:   ${WORLD_SIZE}"
    fi

    echo -e "${BOLD}============================================================${RESET}"
    echo ""
}

# ============================================================================
# Logging helpers
# ============================================================================
_log_file() {
    local label="$1"
    echo "${LOG_DIR}/${label//\//_}.log"
}

record_result() {
    local label="$1"
    local rc="$2"
    local elapsed="$3"
    local log_file="$4"

    ((TOTAL++)) || true

    if [[ $rc -eq 0 ]]; then
        ((PASSED++)) || true
        printf "  ${GREEN}[PASS]${RESET} %-45s ${BOLD}(%.1fs)${RESET}\n" "$label" "$elapsed"
    elif [[ $rc -eq 5 ]]; then
        # pytest exit code 5 = no tests collected (all skipped)
        ((SKIPPED++)) || true
        printf "  ${YELLOW}[SKIP]${RESET} %-45s ${BOLD}(%.1fs)${RESET}\n" "$label" "$elapsed"
    else
        ((FAILED++)) || true
        FAIL_LIST+=("$label")
        printf "  ${RED}[FAIL]${RESET} %-45s ${BOLD}(%.1fs)${RESET}  → %s\n" "$label" "$elapsed" "$log_file"
    fi
}

# ============================================================================
# Test runners
# ============================================================================
# run_pytest LABEL TEST_PATH [--env KEY=VAL ...] [--pytest-args ARG ...]
#
# Everything after --env until --pytest-args (or end) is treated as extra
# environment variables.  Everything after --pytest-args is appended to the
# pytest command line.
run_pytest() {
    local label="$1"; shift
    local test_path="$1"; shift

    # Parse optional --env / --pytest-args sections
    local extra_env=()
    local extra_pytest_args=()
    local mode=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --env)          mode="env"; shift; continue ;;
            --pytest-args)  mode="pytest"; shift; continue ;;
        esac
        if [[ "$mode" == "env" ]]; then
            extra_env+=("$1")
        elif [[ "$mode" == "pytest" ]]; then
            extra_pytest_args+=("$1")
        fi
        shift
    done

    local log_file
    log_file=$(_log_file "$label")

    local cmd_env=(
        "TEST_FASTSAFETENSORS_FRAMEWORK=${FRAMEWORK}"
    )
    if [[ ${#extra_env[@]} -gt 0 ]]; then
        cmd_env+=("${extra_env[@]}")
    fi

    local pytest_args=(-s)
    if $COVERAGE; then
        pytest_args+=(--cov="${LIBDIR}")
        cmd_env+=("COVERAGE_FILE=.coverage_${COVERAGE_IDX}")
        ((COVERAGE_IDX++)) || true
    fi
    if [[ ${#extra_pytest_args[@]} -gt 0 ]]; then
        pytest_args+=("${extra_pytest_args[@]}")
    fi

    local start_time
    start_time=$(python3 -c "import time; print(time.time())")

    local rc=0
    (
        cd "${TESTS_DIR}"
        env "${cmd_env[@]}" python3 -m pytest "${pytest_args[@]}" "${test_path}" \
            > "${log_file}" 2>&1
    ) || rc=$?

    local end_time
    end_time=$(python3 -c "import time; print(time.time())")
    local elapsed
    elapsed=$(python3 -c "print(f'{${end_time} - ${start_time}:.1f}')")

    record_result "$label" "$rc" "$elapsed" "$log_file"
    return 0  # never fail the script; we track results separately
}

run_torchrun() {
    local label="$1"
    local test_file="$2"
    local nnodes="$3"
    local rank="$4"
    local log_file
    log_file=$(_log_file "${label}")

    local cmd_env=(
        "TEST_FASTSAFETENSORS_FRAMEWORK=${FRAMEWORK}"
    )
    local pytest_args=(-s)
    if $COVERAGE; then
        pytest_args+=(--cov="${LIBDIR}")
        cmd_env+=("COVERAGE_FILE=.coverage_${COVERAGE_IDX}")
        ((COVERAGE_IDX++)) || true
    fi

    (
        cd "${TESTS_DIR}"
        env "${cmd_env[@]}" torchrun \
            --nnodes="${nnodes}" \
            --master_addr=0.0.0.0 \
            --master_port="${MASTER_PORT}" \
            --node_rank="${rank}" \
            "${test_file}" "${pytest_args[@]}" "${test_file}" \
            > "${log_file}" 2>&1
    )
}

run_paddle_distributed() {
    local label="$1"
    local test_file="$2"
    local nnodes="$3"
    local rank="$4"
    local log_file
    log_file=$(_log_file "${label}")

    local cmd_env=(
        "TEST_FASTSAFETENSORS_FRAMEWORK=${FRAMEWORK}"
        "PADDLE_DISTRI_BACKEND=gloo"
        "WORLD_SIZE=${nnodes}"
    )
    local pytest_args=(-s)
    if $COVERAGE; then
        pytest_args+=(--cov="${LIBDIR}")
        cmd_env+=("COVERAGE_FILE=.coverage_${COVERAGE_IDX}")
        ((COVERAGE_IDX++)) || true
    fi

    (
        cd "${TESTS_DIR}"
        env "${cmd_env[@]}" python3 -m paddle.distributed.launch \
            --nnodes "${nnodes}" \
            --master "127.0.0.1:${MASTER_PORT}" \
            --rank "${rank}" \
            "${test_file}" "${pytest_args[@]}" "${test_file}" \
            > "${log_file}" 2>&1
    )
}

# ============================================================================
# Test scenes
# ============================================================================
run_mock_tests() {
    echo -e "\n${BOLD}── Mock Tests ──${RESET}"

    run_pytest "mock/test_fastsafetensors" \
        test_fastsafetensors.py

    run_pytest "mock/test_fastsafetensors(no-gpu)" \
        test_fastsafetensors.py \
        --env "CUDA_VISIBLE_DEVICES="

    run_pytest "mock/test_parallel" \
        test_parallel.py \
        --pytest-args -v

    run_pytest "mock/test_vllm" \
        test_vllm.py
}

run_3fs_tests() {
    echo -e "\n${BOLD}── 3FS Tests ──${RESET}"

    if ! $HAS_3FS; then
        echo -e "  ${YELLOW}[SKIP]${RESET} 3FS not available — skipping all 3FS tests"
        ((TOTAL += 3)) || true
        ((SKIPPED += 3)) || true
        return
    fi

    run_pytest "3fs/test_threefs" \
        threefs/test_threefs.py \
        --pytest-args -v

    run_pytest "3fs/test_parallel_threefs" \
        threefs/test_parallel_threefs.py \
        --pytest-args -v

    run_pytest "3fs/test_parallel(with-3fs)" \
        test_parallel.py \
        --pytest-args -v
}

run_distributed_tests() {
    echo -e "\n${BOLD}── Distributed Tests ──${RESET}"

    # --- Single-node ---
    if [[ "$FRAMEWORK" == "pytorch" ]]; then
        local label="distributed/test_multi-1node"
        local log_file
        log_file=$(_log_file "$label")

        local start_time
        start_time=$(python3 -c "import time; print(time.time())")

        local rc=0
        run_torchrun "$label" test_multi.py 1 0 || rc=$?

        local end_time
        end_time=$(python3 -c "import time; print(time.time())")
        local elapsed
        elapsed=$(python3 -c "print(f'{${end_time} - ${start_time}:.1f}')")
        record_result "$label" "$rc" "$elapsed" "$log_file"

        # --- Multi-node ---
        label="distributed/test_multi-${WORLD_SIZE}node"
        log_file=$(_log_file "$label")
        start_time=$(python3 -c "import time; print(time.time())")

        local pids=()
        rc=0
        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            run_torchrun "${label}-rank${rank}" test_multi.py "${WORLD_SIZE}" "${rank}" &
            pids+=($!)
        done

        # Wait for all ranks
        for pid in "${pids[@]}"; do
            wait "$pid" || rc=$?
        done

        end_time=$(python3 -c "import time; print(time.time())")
        elapsed=$(python3 -c "print(f'{${end_time} - ${start_time}:.1f}')")
        record_result "$label" "$rc" "$elapsed" "$log_file"

    elif [[ "$FRAMEWORK" == "paddle" ]]; then
        local label="distributed/test_multi-paddle-${WORLD_SIZE}node"
        local log_file
        log_file=$(_log_file "$label")

        local start_time
        start_time=$(python3 -c "import time; print(time.time())")

        local pids=()
        local rc=0
        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            run_paddle_distributed "${label}-rank${rank}" test_multi.py "${WORLD_SIZE}" "${rank}" &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid" || rc=$?
        done

        local end_time
        end_time=$(python3 -c "import time; print(time.time())")
        local elapsed
        elapsed=$(python3 -c "print(f'{${end_time} - ${start_time}:.1f}')")
        record_result "$label" "$rc" "$elapsed" "$log_file"
    fi
}

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    echo ""
    echo -e "${BOLD}============================================================${RESET}"
    echo -e "${BOLD}  Test Summary${RESET}"
    echo -e "${BOLD}============================================================${RESET}"
    echo -e "  Total:    ${TOTAL}"
    echo -e "  Passed:   ${GREEN}${PASSED}${RESET}"
    echo -e "  Failed:   ${RED}${FAILED}${RESET}"
    echo -e "  Skipped:  ${YELLOW}${SKIPPED}${RESET}"

    if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
        echo ""
        echo -e "  ${RED}Failed tests:${RESET}"
        for f in "${FAIL_LIST[@]}"; do
            echo -e "    - ${f}"
        done
    fi

    echo -e "${BOLD}============================================================${RESET}"
    echo -e "  Logs: ${LOG_DIR}/"

    if $COVERAGE; then
        echo -e "  Coverage: htmlcov/index.html"
    fi

    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    parse_args "$@"

    mkdir -p "${LOG_DIR}"

    check_environment

    # Run selected scenes
    $RUN_MOCK && run_mock_tests
    $RUN_3FS && run_3fs_tests
    $RUN_DISTRIBUTED && run_distributed_tests

    # Coverage
    if $COVERAGE; then
        echo -e "\n${BOLD}── Coverage ──${RESET}"
        (
            cd "${TESTS_DIR}"
            coverage combine .coverage_* 2>/dev/null || true
            coverage html 2>/dev/null || true
        )
        echo -e "  ${GREEN}Coverage report generated: htmlcov/index.html${RESET}"
    fi

    print_summary

    # Exit code
    if [[ $FAILED -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
