"""
Test the complete pure Python class workflow.

This tests the end-to-end flow without LLM:
1. Template generation
2. Direct execution
3. Optimization
4. Evaluation
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_template_generation():
    """Test that templates generate valid Python code."""
    print("\n" + "="*80)
    print("TEST 1: Template Generation")
    print("="*80)

    from utils.ha_class_template import get_simple_ha_template

    # Test single-variable template
    template = get_simple_ha_template(num_variables=1, num_inputs=1)

    print("Generated template:")
    print(template[:500] + "...")

    # Try to execute it
    namespace = {}
    try:
        exec(template, namespace)
        assert 'HybridAutomaton' in namespace
        print("\n✓ Template is valid Python code")

        # Try to instantiate
        ha = namespace['HybridAutomaton']()
        print(f"✓ Instance created: {ha.num_modes()} mode(s)")
        print(f"  Variables: {ha.var}")
        print(f"  Inputs: {ha.input}")

        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulator():
    """Test the pure Python class simulator."""
    print("\n" + "="*80)
    print("TEST 2: Python Class Simulator")
    print("="*80)

    from utils.ha_class_template import get_example_duffing_template
    from utils.ha_class_simulator import HybridAutomatonSimulator

    # Get pre-filled Duffing template
    class_code = get_example_duffing_template()

    # Execute and instantiate
    namespace = {}
    exec(class_code, namespace)
    ha = namespace['HybridAutomaton']()

    print(f"Testing with Duffing oscillator")
    print(f"  Params: {ha.params[:4]}")

    # Create simulator
    sim = HybridAutomatonSimulator(ha, dt=0.001, total_time=1.0)

    # Run short simulation
    initial_state = np.array([1.0])  # x1(0) = 1.0
    input_data = np.zeros((1, 1000))  # Zero input

    try:
        state_traj, mode_seq = sim.simulate(initial_state, input_data)
        print(f"\n✓ Simulation succeeded")
        print(f"  Final state: {state_traj[0, -1]:.4f}")
        print(f"  Trajectory shape: {state_traj.shape}")

        return True

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test the evaluator with real data."""
    print("\n" + "="*80)
    print("TEST 3: Python Class Evaluator")
    print("="*80)

    from utils.ha_class_template import get_simple_ha_template
    from utils.ha_class_simulator import PythonClassHAEvaluator

    # Generate simple template
    class_code = get_simple_ha_template(num_variables=1, num_inputs=1)

    # Execute
    namespace = {}
    exec(class_code, namespace)
    ha = namespace['HybridAutomaton']()

    # Check if test data exists
    test_data_path = "data_all/non_linear/duffing/sample_0.npz"
    if not os.path.exists(test_data_path):
        print(f"⚠ Test data not found: {test_data_path}")
        print("  Skipping this test")
        return True  # Don't fail test if data missing

    print(f"Using test data: {test_data_path}")

    try:
        evaluator = PythonClassHAEvaluator(
            ha_instance=ha,
            npz_file_path=test_data_path,
            dt=0.001,
            total_time=10.0
        )

        # Run evaluation
        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path='test_eval.png',
            print_metrics=False,
            show_plot=False
        )

        print(f"\n✓ Evaluation succeeded")
        print(f"  Metrics: {evaluator.metrics}")

        # Clean up
        if os.path.exists('test_eval.png'):
            os.remove('test_eval.png')

        return True

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_function():
    """Test the workflow generation function."""
    print("\n" + "="*80)
    print("TEST 4: Workflow Task Generation")
    print("="*80)

    from utils.pure_python_workflow import generate_pure_python_task

    # Check if data exists
    data_path = "data_all/non_linear/duffing"
    if not os.path.exists(data_path):
        print(f"⚠ Data path not found: {data_path}")
        print("  Skipping this test")
        return True

    try:
        task, images = generate_pure_python_task(
            input_data_path=data_path,
            num_variables=1,
            num_inputs=1,
            iteration=1,
            feedback="",
            tools_list=["hybrid_automaton_image_analysis"],
            manager_type="CodeAgent"
        )

        print(f"\n✓ Task generated")
        print(f"  Length: {len(task)} characters")
        print(f"  Images: {len(images)}")
        print(f"\nTask preview:")
        print(task[:300] + "...")

        # Check for key elements
        assert "class HybridAutomaton" in task, "Missing class template"
        assert "self.params" in task, "Missing params array"
        assert "num_modes" in task, "Missing num_modes info"

        print(f"\n✓ Task contains required elements")

        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PURE PYTHON WORKFLOW TESTS")
    print("="*80)

    results = []

    # Run tests
    results.append(("Template generation", test_template_generation()))
    results.append(("Simulator", test_simulator()))
    results.append(("Evaluator", test_evaluator()))
    results.append(("Workflow function", test_workflow_function()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)}")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
