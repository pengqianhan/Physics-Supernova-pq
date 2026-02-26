"""
Test HybridAutomatonImageTool with dual backends:
  - Gemini models via native genai.Client (with code_execution)
  - Kimi-k2.5 via litellm

Tests both <image_N> (markdown trace) and <iter_image_X_Y> (registered overlay) placeholders.
Covers ball (simple) and duffing (complex) datasets as required by CLAUDE.md guideline #5.

Usage:
    python testfiles/test_imgTools_ha_dual_backend.py               # run all tests
    python testfiles/test_imgTools_ha_dual_backend.py --gemini-only  # Gemini tests only
    python testfiles/test_imgTools_ha_dual_backend.py --kimi-only    # Kimi tests only
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils.imgTools_ha import HybridAutomatonImageTool
from utils.markdown_utils import load_trace_data_from_filepath


# ---------------------------------------------------------------------------
# Test image paths (ball = simple, duffing = complex)
# ---------------------------------------------------------------------------
BALL_OVERLAY = "evaluation_results/ATVA/ball/runs/20260225_143306_7ba7/iter_1/overlay_0.png"
DUFFING_OVERLAY = "evaluation_results/non_linear/duffing/runs/20260225_160850_4139/best_iter_5/overlay_0.png"

BALL_TRACE_DIR = "data_all/ATVA/ball"
DUFFING_TRACE_DIR = "data_all/non_linear/duffing"

# Model IDs
GEMINI_MODEL = "gemini-3-flash-preview"
KIMI_MODEL = "openai/kimi-k2.5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockWorkerAgent:
    """Mock agent that provides markdown_content_high_res_image for <image_N> placeholders."""
    def __init__(self, trace_dir: str):
        self.markdown_content_high_res_image = load_trace_data_from_filepath(
            str(project_root / trace_dir)
        )


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_response(response: str, test_name: str) -> bool:
    """Verify that the response is a meaningful non-error string."""
    if response.startswith("Error:"):
        print(f"  [FAIL] {test_name}")
        print(f"         Got error: {response}")
        return False
    if response == "No response":
        print(f"  [FAIL] {test_name}")
        print(f"         Model returned no response")
        return False
    # Truncate long responses for display
    display = response[:200] + "..." if len(response) > 200 else response
    print(f"  [PASS] {test_name}")
    print(f"         Response: {display}")
    return True


def check_env(model_id: str) -> bool:
    """Verify required API key is set for the given model."""
    if model_id.startswith("gemini") or model_id.startswith("gemini/"):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            print(f"  [SKIP] GEMINI_API_KEY not set")
            return False
    elif "kimi" in model_id or "moonshot" in model_id:
        key = os.environ.get("MOONSHOT_API_KEY")
        if not key:
            print(f"  [SKIP] MOONSHOT_API_KEY not set")
            return False
    return True


# ---------------------------------------------------------------------------
# Test: _is_gemini_model routing
# ---------------------------------------------------------------------------
def test_model_routing():
    """Unit test: verify _is_gemini_model correctly classifies model IDs."""
    separator("Test: Model routing logic (_is_gemini_model)")

    cases = [
        ("gemini-3-flash-preview", True),
        ("gemini/gemini-flash-lite-latest", True),
        ("gemini-2.5-flash-lite", True),
        ("openai/kimi-k2.5", False),
        ("moonshot/moonshot-v1-8k", False),
        ("openai/moonshot-v1-8k", False),
        ("gpt-4o", False),
    ]

    passed = 0
    for model_id, expected in cases:
        tool = HybridAutomatonImageTool(vision_model_id=model_id)
        result = tool._is_gemini_model()
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {model_id:40s} -> is_gemini={result} (expected {expected})")

    print(f"\n  Routing: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ---------------------------------------------------------------------------
# Test: _get_litellm_kwargs returns correct routing info
# ---------------------------------------------------------------------------
def test_litellm_kwargs():
    """Unit test: verify _get_litellm_kwargs returns correct keys for each provider."""
    separator("Test: LiteLLM kwargs routing (_get_litellm_kwargs)")

    cases = [
        ("openai/kimi-k2.5", "MOONSHOT_API_KEY", "api.moonshot.cn"),
        ("kimi-k2.5", "MOONSHOT_API_KEY", "api.moonshot.cn"),
        ("moonshot/moonshot-v1-8k", "MOONSHOT_API_KEY", "api.moonshot.cn"),
        ("gemini/gemini-flash-lite-latest", "GEMINI_API_KEY", None),
        ("gpt-4o", None, None),
    ]

    passed = 0
    for model_id, expected_key_env, expected_base_substr in cases:
        tool = HybridAutomatonImageTool(vision_model_id=model_id)
        kwargs = tool._get_litellm_kwargs()

        ok = True
        if expected_key_env:
            if "api_key" not in kwargs:
                ok = False
        else:
            if "api_key" in kwargs:
                ok = False

        if expected_base_substr:
            if "api_base" not in kwargs or expected_base_substr not in kwargs.get("api_base", ""):
                ok = False

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        keys_summary = {k: ("***" if k == "api_key" else v) for k, v in kwargs.items()}
        print(f"  [{status}] {model_id:40s} -> {keys_summary}")

    print(f"\n  LiteLLM kwargs: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ---------------------------------------------------------------------------
# Live API tests: Gemini backend
# ---------------------------------------------------------------------------
def test_gemini_iter_image(dataset: str, overlay_path: str):
    """Test Gemini backend with a registered overlay image (<iter_image_X_Y>)."""
    test_name = f"Gemini iter_image ({Path(overlay_path).parts[-4]})"
    if not check_env(GEMINI_MODEL):
        return False

    abs_overlay = str(project_root / overlay_path)
    if not Path(abs_overlay).is_file():
        print(f"  [SKIP] {test_name}: overlay not found at {overlay_path}")
        return False

    tool = HybridAutomatonImageTool(vision_model_id=GEMINI_MODEL)
    success, err = tool.register_iteration_image(100, abs_overlay)
    if not success:
        print(f"  [FAIL] {test_name}: registration failed: {err}")
        return False

    response = tool.forward("<iter_image_1_0>", "Describe what this overlay plot shows. What system is being modeled?")
    return check_response(response, test_name)


def test_gemini_markdown_image(trace_dir: str, dataset_label: str):
    """Test Gemini backend with markdown trace image (<image_0>)."""
    test_name = f"Gemini markdown <image_0> ({dataset_label})"
    if not check_env(GEMINI_MODEL):
        return False

    mock_agent = MockWorkerAgent(trace_dir)
    tool = HybridAutomatonImageTool(worker_agent=mock_agent, vision_model_id=GEMINI_MODEL)
    response = tool.forward("<image_0>", "Describe the trajectory shown in this plot. How many states are visible?")
    return check_response(response, test_name)


# ---------------------------------------------------------------------------
# Live API tests: Kimi backend (via litellm)
# ---------------------------------------------------------------------------
def test_kimi_iter_image(dataset: str, overlay_path: str):
    """Test Kimi-k2.5 backend with a registered overlay image (<iter_image_X_Y>)."""
    test_name = f"Kimi iter_image ({Path(overlay_path).parts[-4]})"
    if not check_env(KIMI_MODEL):
        return False

    abs_overlay = str(project_root / overlay_path)
    if not Path(abs_overlay).is_file():
        print(f"  [SKIP] {test_name}: overlay not found at {overlay_path}")
        return False

    tool = HybridAutomatonImageTool(vision_model_id=KIMI_MODEL)
    success, err = tool.register_iteration_image(100, abs_overlay)
    if not success:
        print(f"  [FAIL] {test_name}: registration failed: {err}")
        return False

    response = tool.forward("<iter_image_1_0>", "Describe what this overlay plot shows. What system is being modeled?")
    return check_response(response, test_name)


def test_kimi_markdown_image(trace_dir: str, dataset_label: str):
    """Test Kimi-k2.5 backend with markdown trace image (<image_0>)."""
    test_name = f"Kimi markdown <image_0> ({dataset_label})"
    if not check_env(KIMI_MODEL):
        return False

    mock_agent = MockWorkerAgent(trace_dir)
    tool = HybridAutomatonImageTool(worker_agent=mock_agent, vision_model_id=KIMI_MODEL)
    response = tool.forward("<image_0>", "Describe the trajectory shown in this plot. How many states are visible?")
    return check_response(response, test_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test HybridAutomatonImageTool dual backends")
    parser.add_argument("--gemini-only", action="store_true", help="Run Gemini tests only")
    parser.add_argument("--kimi-only", action="store_true", help="Run Kimi tests only")
    args = parser.parse_args()

    run_gemini = not args.kimi_only
    run_kimi = not args.gemini_only

    results = []

    # --- Unit tests (always run) ---
    separator("UNIT TESTS")
    results.append(("model_routing", test_model_routing()))
    results.append(("litellm_kwargs", test_litellm_kwargs()))

    # --- Gemini live API tests ---
    if run_gemini:
        separator("GEMINI LIVE API TESTS")

        # Ball (simple)
        results.append(("gemini_iter_ball", test_gemini_iter_image("ball", BALL_OVERLAY)))
        time.sleep(1)
        results.append(("gemini_md_ball", test_gemini_markdown_image(BALL_TRACE_DIR, "ball")))
        time.sleep(1)

        # Duffing (complex)
        results.append(("gemini_iter_duffing", test_gemini_iter_image("duffing", DUFFING_OVERLAY)))
        time.sleep(1)
        results.append(("gemini_md_duffing", test_gemini_markdown_image(DUFFING_TRACE_DIR, "duffing")))

    # --- Kimi live API tests ---
    if run_kimi:
        separator("KIMI-K2.5 LIVE API TESTS (via litellm)")

        # Ball (simple)
        time.sleep(2)  # Rate limit protection between providers
        results.append(("kimi_iter_ball", test_kimi_iter_image("ball", BALL_OVERLAY)))
        time.sleep(3)  # Kimi rate limits are stricter
        results.append(("kimi_md_ball", test_kimi_markdown_image(BALL_TRACE_DIR, "ball")))
        time.sleep(3)

        # Duffing (complex)
        results.append(("kimi_iter_duffing", test_kimi_iter_image("duffing", DUFFING_OVERLAY)))
        time.sleep(3)
        results.append(("kimi_md_duffing", test_kimi_markdown_image(DUFFING_TRACE_DIR, "duffing")))

    # --- Summary ---
    separator("SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Total: {passed}/{total} passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
