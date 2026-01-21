"""Test for gen_summary function in run_llm_ha_gamma.py"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_llm_ha_gamma import gen_summary


def test_gen_summary_with_image():
    """Test gen_summary with valid metrics, HA spec, and plot image."""
    print("\n" + "=" * 60)
    print("Test 1: gen_summary with image")
    print("=" * 60)
    
    # Sample metrics dict
    metrics_dict = {
        "mean_diff": 0.00523,
        "max_diff": 0.0234,
        "tc": 0.012,
        "clustering_error": 0.008
    }
    
    # Sample HA specification
    ha_specification = {
        "automaton": {
            "var": "x1",
            "input": "u1",
            "mode": [
                {
                    "id": 1,
                    "eq": "x1[2] = -0.1*x1[1] - x1[0] - x1[0]**3 + u1"
                }
            ],
            "edge": []
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "order": 2,
            "need_reset": False,
            "non_linear_items": ""
        }
    }
    
    # Use an existing sample image from data_all
    plot_path = "data_all/FaMoS/complex_tank/sample_0.png"
    
    # Check if the image exists
    if not os.path.isfile(plot_path):
        print(f"Warning: Test image not found at {plot_path}")
        print("Will test without image...")
        plot_path = None
    
    # Call gen_summary
    summary = gen_summary(
        metrics_dict=metrics_dict,
        ha_specification=ha_specification,
        plot_path=plot_path,
        model_id="gemini/gemini-2.5-flash-lite"  # Use a cheaper model for testing
    )
    
    print(f"\nGenerated Summary:\n{summary}")
    print(f"\nSummary length: {len(summary)} characters")
    
    assert summary is not None, "Summary should not be None"
    assert len(summary) > 0, "Summary should not be empty"
    assert "failed" not in summary.lower() or "LLM summary generation failed" not in summary, \
        "Summary generation should succeed"
    
    print("\n✓ Test 1 PASSED")
    return summary


def test_gen_summary_without_image():
    """Test gen_summary without a plot image."""
    print("\n" + "=" * 60)
    print("Test 2: gen_summary without image")
    print("=" * 60)
    
    metrics_dict = {
        "mean_diff": 0.0123,
        "max_diff": 0.0567,
        "tc": 0.025
    }
    
    ha_specification = {
        "automaton": {
            "var": "x1, x2",
            "input": "",
            "mode": [
                {
                    "id": 1,
                    "eq": "x1[1] = x2[0], x2[1] = -x1[0] - 0.5*x2[0]"
                }
            ],
            "edge": []
        },
        "config": {
            "dt": 0.001,
            "total_time": 5.0,
            "order": 1
        }
    }
    
    # No image
    plot_path = None
    
    summary = gen_summary(
        metrics_dict=metrics_dict,
        ha_specification=ha_specification,
        plot_path=plot_path,
        model_id="gemini/gemini-2.5-flash-lite"
    )
    
    print(f"\nGenerated Summary:\n{summary}")
    
    assert summary is not None, "Summary should not be None"
    assert len(summary) > 0, "Summary should not be empty"
    
    print("\n✓ Test 2 PASSED")
    return summary


def test_gen_summary_empty_metrics():
    """Test gen_summary with empty metrics dict."""
    print("\n" + "=" * 60)
    print("Test 3: gen_summary with empty metrics")
    print("=" * 60)
    
    metrics_dict = {}
    
    ha_specification = {
        "automaton": {"var": "x1", "input": "", "mode": [], "edge": []},
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    
    summary = gen_summary(
        metrics_dict=metrics_dict,
        ha_specification=ha_specification,
        plot_path=None,
        model_id="gemini/gemini-2.5-flash-lite"
    )
    
    print(f"\nGenerated Summary:\n{summary}")
    
    # With empty metrics, function should return early with fallback message
    assert summary == "No metrics available for analysis.", \
        f"Expected fallback message for empty metrics, got: {summary}"
    
    print("\n✓ Test 3 PASSED")
    return summary


def test_gen_summary_invalid_image_path():
    """Test gen_summary with invalid image path."""
    print("\n" + "=" * 60)
    print("Test 4: gen_summary with invalid image path")
    print("=" * 60)
    
    metrics_dict = {
        "mean_diff": 0.001,
        "max_diff": 0.005
    }
    
    ha_specification = {
        "automaton": {"var": "x1", "input": "u1", "mode": [{"id": 1, "eq": "x1[1] = u1"}], "edge": []},
        "config": {"dt": 0.001, "total_time": 10.0}
    }
    
    # Invalid path
    plot_path = "/nonexistent/path/to/image.png"
    
    summary = gen_summary(
        metrics_dict=metrics_dict,
        ha_specification=ha_specification,
        plot_path=plot_path,
        model_id="gemini/gemini-2.5-flash-lite"
    )
    
    print(f"\nGenerated Summary:\n{summary}")
    
    # Should still generate a summary (without image analysis)
    assert summary is not None, "Summary should not be None"
    assert len(summary) > 0, "Summary should not be empty"
    
    print("\n✓ Test 4 PASSED")
    return summary


def test_gen_summary_multimode_ha():
    """Test gen_summary with a multi-mode HA specification."""
    print("\n" + "=" * 60)
    print("Test 5: gen_summary with multi-mode HA")
    print("=" * 60)
    
    metrics_dict = {
        "mean_diff": 0.0089,
        "max_diff": 0.0321,
        "tc": 0.015,
        "clustering_error": 0.005
    }
    
    # Multi-mode HA with edges
    ha_specification = {
        "automaton": {
            "var": "x1, x2",
            "input": "u1",
            "mode": [
                {
                    "id": 1,
                    "eq": "x1[1] = x2[0], x2[1] = -x1[0] + u1"
                },
                {
                    "id": 2,
                    "eq": "x1[1] = x2[0], x2[1] = -2*x1[0] - x2[0] + u1"
                }
            ],
            "edge": [
                {
                    "direction": "1 -> 2",
                    "condition": "x1 >= 1.0"
                },
                {
                    "direction": "2 -> 1",
                    "condition": "x1 <= -1.0"
                }
            ]
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "order": 1,
            "need_reset": False
        }
    }
    
    # Use sample image if available
    plot_path = "data_all/ATVA/ball/sample_0.png"
    if not os.path.isfile(plot_path):
        plot_path = None
    
    summary = gen_summary(
        metrics_dict=metrics_dict,
        ha_specification=ha_specification,
        plot_path=plot_path,
        model_id="gemini/gemini-2.5-flash-lite"
    )
    
    print(f"\nGenerated Summary:\n{summary}")
    
    assert summary is not None, "Summary should not be None"
    assert len(summary) > 0, "Summary should not be empty"
    
    print("\n✓ Test 5 PASSED")
    return summary


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Running all gen_summary tests")
    print("=" * 70)
    
    # Ensure GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set it before running tests:")
        print("  export GEMINI_API_KEY=your_api_key_here")
        return False
    
    tests = [
        ("Test 1: with image", test_gen_summary_with_image),
        ("Test 2: without image", test_gen_summary_without_image),
        ("Test 3: empty metrics", test_gen_summary_empty_metrics),
        ("Test 4: invalid image path", test_gen_summary_invalid_image_path),
        ("Test 5: multi-mode HA", test_gen_summary_multimode_ha),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} FAILED: {e}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test gen_summary function")
    parser.add_argument("--test", type=int, default=None, 
                        help="Run specific test (1-5), or all if not specified")
    args = parser.parse_args()
    
    if args.test is None:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    elif args.test == 1:
        test_gen_summary_with_image()
    elif args.test == 2:
        test_gen_summary_without_image()
    elif args.test == 3:
        test_gen_summary_empty_metrics()
    elif args.test == 4:
        test_gen_summary_invalid_image_path()
    elif args.test == 5:
        test_gen_summary_multimode_ha()
    else:
        print(f"Invalid test number: {args.test}. Valid options: 1-5")
        sys.exit(1)
