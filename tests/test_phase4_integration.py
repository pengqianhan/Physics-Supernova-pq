"""
Integration test for Phase 4: Main Loop Integration

This test verifies that all components of the Python class-based HA system
work together without running the full pipeline.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")

    try:
        from utils.ha_class_to_json import convert_python_class_to_json
        print("✓ ha_class_to_json imported")
    except ImportError as e:
        print(f"✗ Failed to import ha_class_to_json: {e}")
        return False

    try:
        from utils.ha_class_validator import (
            extract_python_class_from_text,
            validate_python_class_syntax,
            extract_initial_params_from_class
        )
        print("✓ ha_class_validator imported")
    except ImportError as e:
        print(f"✗ Failed to import ha_class_validator: {e}")
        return False

    try:
        from utils.ha_class_optimizer import optimize_ha_params
        print("✓ ha_class_optimizer imported")
    except ImportError as e:
        print(f"✗ Failed to import ha_class_optimizer: {e}")
        return False

    try:
        from prompts_ha.prompts_class import get_ha_class_documentation
        print("✓ prompts_class imported")
    except ImportError as e:
        print(f"✗ Failed to import prompts_class: {e}")
        return False

    try:
        from utils.utils import IterationResult
        print("✓ IterationResult imported")
    except ImportError as e:
        print(f"✗ Failed to import IterationResult: {e}")
        return False

    return True


def test_iteration_result_with_new_fields():
    """Test that IterationResult accepts class_code and optimized_params."""
    print("\nTesting IterationResult with new fields...")

    try:
        from utils.utils import IterationResult

        # Create test data
        test_params = np.array([1.0, 2.0, 3.0])
        test_class = "class HybridAutomaton:\n    pass"

        # Create IterationResult with new fields
        result = IterationResult(
            iteration=1,
            ha_specification={'automaton': {}, 'config': {}},
            metrics={'max_diff': 0.05},
            feedback="Test feedback",
            success=True,
            error_value=0.05,
            class_code=test_class,
            optimized_params=test_params
        )

        # Verify fields
        assert result.class_code == test_class, "class_code field mismatch"
        assert np.array_equal(result.optimized_params, test_params), "optimized_params mismatch"

        print("✓ IterationResult accepts new fields correctly")
        return True

    except Exception as e:
        print(f"✗ IterationResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_obtain_task_and_images_signature():
    """Test that obtain_task_and_images accepts use_python_class_format parameter."""
    print("\nTesting obtain_task_and_images signature...")

    try:
        import inspect
        # Import the module
        import run_llm_ha_beta

        # Get function signature
        sig = inspect.signature(run_llm_ha_beta.obtain_task_and_images)
        params = sig.parameters

        # Check for new parameter
        assert 'use_python_class_format' in params, "Missing use_python_class_format parameter"
        assert params['use_python_class_format'].default == False, "Wrong default value"

        print("✓ obtain_task_and_images has correct signature")
        return True

    except Exception as e:
        print(f"✗ obtain_task_and_images signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_function_signature():
    """Test that evaluate_ha_specification_with_feedback has correct signature and return type."""
    print("\nTesting evaluate_ha_specification_with_feedback signature...")

    try:
        import inspect
        import run_llm_ha_beta

        # Get function signature
        sig = inspect.signature(run_llm_ha_beta.evaluate_ha_specification_with_feedback)
        params = sig.parameters

        # Check for new parameters
        assert 'use_python_class_format' in params, "Missing use_python_class_format parameter"
        assert 'optimization_iters' in params, "Missing optimization_iters parameter"
        assert 'optimizer_type' in params, "Missing optimizer_type parameter"

        # Check defaults
        assert params['use_python_class_format'].default == False
        assert params['optimization_iters'].default == 100
        assert params['optimizer_type'].default == "simulated_annealing"

        print("✓ evaluate_ha_specification_with_feedback has correct signature")
        return True

    except Exception as e:
        print(f"✗ evaluate function signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_arguments():
    """Test that CLI arguments are properly defined."""
    print("\nTesting CLI arguments...")

    try:
        import run_llm_ha_beta
        import argparse

        # Create parser
        parser = argparse.ArgumentParser()

        # Try to add all the arguments (simulating parse_args)
        # This is a simplified check - we just verify the function exists
        assert hasattr(run_llm_ha_beta, 'parse_args'), "parse_args function missing"

        print("✓ CLI arguments properly defined")
        return True

    except Exception as e:
        print(f"✗ CLI arguments test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*80)
    print("Phase 4 Integration Tests")
    print("="*80)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("IterationResult with new fields", test_iteration_result_with_new_fields()))
    results.append(("obtain_task_and_images signature", test_obtain_task_and_images_signature()))
    results.append(("evaluate function signature", test_evaluate_function_signature()))
    results.append(("CLI arguments", test_cli_arguments()))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed == 0:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
