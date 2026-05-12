# ABOUTME: CLI entrypoint for moulinette - generates exercises and grades submissions.
# ABOUTME: Usage: uv run python -m moulinette prepare_exercises|grade_student_answers

import json
import shutil
from pathlib import Path

import fire

from moulinette.extract_functions_infos import generate_function_calling_definition
from moulinette.generate_tests_and_corrections import (
    save_function_calling_corrections,
    save_function_calling_tests,
)
from moulinette.functions_definition import get_exercises_by_visibility
from moulinette.output_formatter import ColoredOutput


class Moulinette:
    """CLI for generating function calling exercises and grading student submissions."""

    def __init__(self):
        self.output = ColoredOutput()

    def prepare_exercises(
        self,
        output: str = "data",
        set: str = "public",
    ) -> None:
        """Generate exercise files for students.

        Args:
            output: Output directory path (default: "data")
            set: Exercise set to generate - "public" or "private" (default: "public")
        """
        if set not in ("public", "private"):
            self.output.error(f"Invalid set '{set}'. Must be 'public' or 'private'.")
            return

        output_path = Path(output)
        input_dir = output_path / "input"
        correction_dir = output_path / "correction"

        # Clean and recreate directories
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(correction_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        correction_dir.mkdir(parents=True, exist_ok=True)

        functions_definition_path = input_dir / "functions_definition.json"
        function_calling_tests_path = input_dir / "function_calling_tests.json"
        function_calling_corrections_path = correction_dir / "function_calling_corrections.json"

        self.output.info(f"Generating {set} exercise set...")
        self.output.separator()

        # Generate function definitions
        generate_function_calling_definition(
            output_json_path=str(functions_definition_path),
            visibility=set,
        )
        self.output.success(f"Created: {functions_definition_path}")

        # Generate test prompts (for students)
        save_function_calling_tests(
            output_json_path=str(function_calling_tests_path),
            visibility=set,
        )
        self.output.success(f"Created: {function_calling_tests_path}")

        # Generate corrections (for grading)
        save_function_calling_corrections(
            output_json_path=str(function_calling_corrections_path),
            visibility=set,
        )
        self.output.success(f"Created: {function_calling_corrections_path}")

        self.output.separator()
        self.output.info(f"Exercise files generated in: {output_path}")

    def grade_student_answers(
        self,
        student_answer_path: str,
        set: str = "public",
    ) -> None:
        """Grade student submission against corrections.

        Args:
            student_answer_path: Path to the student's function_calls.json file
            set: Exercise set to grade against - "public" or "private" (default: "public")
        """
        if set not in ("public", "private"):
            self.output.error(f"Invalid set '{set}'. Must be 'public' or 'private'.")
            return

        student_path = Path(student_answer_path)
        if not student_path.exists():
            self.output.error(f"Student answer file not found: {student_path}")
            return

        # Load student answers
        with open(student_path) as f:
            student_answers = json.load(f)

        # Get exercises and build function lookup
        filtered_exercises = get_exercises_by_visibility(set)
        fn_name_to_function = {
            fn.__name__: fn for fn in filtered_exercises.keys()
        }

        # Build corrections from exercises
        corrections = []
        for fn, data in filtered_exercises.items():
            for test in data["tests"]:
                corrections.append({
                    "prompt": test["prompt"],
                    "name": fn.__name__,
                    "parameters": test["fn_args"],
                    "expected_output": fn(**test["fn_args"]),
                })

        total_score = 0
        total_tests = len(corrections)

        self.output.info(f"Grading against {set} set ({total_tests} tests)")
        self.output.separator()

        for i, (student_answer, correction) in enumerate(zip(student_answers, corrections), 1):
            self.output.test_header(i, total_tests)
            self.output.prompt(correction["prompt"])

            # Check prompt match
            if student_answer.get("prompt") != correction["prompt"]:
                self.output.expected("Expected prompt", correction["prompt"])
                self.output.actual("Student prompt", student_answer.get("prompt", "<missing>"))
                self.output.test_result(False, "prompt mismatch")
                continue

            fn_name = student_answer.get("name")
            fn_params = student_answer.get("parameters", {})

            # Check function name
            if fn_name not in fn_name_to_function:
                self.output.expected("Expected function", correction["name"])
                self.output.actual("Student called", fn_name)
                self.output.test_result(False, "unknown function")
                continue

            # Try to call the function
            fn = fn_name_to_function[fn_name]
            try:
                student_output = fn(**fn_params)
            except Exception as e:
                self.output.expected("Expected parameters", correction["parameters"])
                self.output.actual("Student parameters", fn_params)
                self.output.error(f"  Error: {e}")
                self.output.test_result(False, "invalid parameters")
                continue

            # Check output
            if student_output != correction["expected_output"]:
                # Show what the student did
                self.output.actual("Student called", f"{fn_name}({fn_params})")
                self.output.actual("Student result", student_output)
                # Show what was expected
                self.output.expected("Expected call", f"{correction['name']}({correction['parameters']})")
                self.output.expected("Expected result", correction["expected_output"])
                self.output.test_result(False, "wrong output")
                continue

            total_score += 1
            self.output.test_result(True)

        self.output.summary(total_score, total_tests)


if __name__ == "__main__":
    fire.Fire(Moulinette)
