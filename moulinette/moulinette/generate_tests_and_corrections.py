# ABOUTME: Generates test prompts and correction files for grading.
# ABOUTME: Creates JSON files that students use as input and moulinette uses for validation.

import json
from typing import Any, List, Callable

from pydantic import BaseModel

from moulinette.functions_definition import exercises, get_exercises_by_visibility


class Correction(BaseModel):
    prompt: str
    name: str
    parameters: dict
    expected_output: Any


def generate_function_calling_corrections(
    exercises_dict: dict[Callable, dict[str, Any]],
) -> List[Correction]:
    """Generate correction objects from exercises dictionary.

    Args:
        exercises_dict: Dictionary of functions to their exercise metadata

    Returns:
        List of Correction objects with expected outputs computed
    """
    corrections = []
    for fn_to_call, data in exercises_dict.items():
        for test in data["tests"]:
            correction = Correction(
                prompt=test["prompt"],
                name=fn_to_call.__name__,
                parameters=test["fn_args"],
                expected_output=fn_to_call(**test["fn_args"])
            )
            corrections.append(correction)
    return corrections


def save_function_calling_corrections(
    output_json_path: str,
    visibility: str = "public",
) -> None:
    """Save corrections JSON file for a given visibility set.

    Args:
        output_json_path: Path where the corrections JSON will be written
        visibility: Either "public" or "private" to filter exercises
    """
    filtered_exercises = get_exercises_by_visibility(visibility)
    corrections = generate_function_calling_corrections(filtered_exercises)
    corrections_list = [correction.model_dump() for correction in corrections]

    with open(output_json_path, "w") as f:
        json.dump(corrections_list, f, indent=2)


def save_function_calling_tests(
    output_json_path: str,
    visibility: str = "public",
) -> None:
    """Save test prompts JSON file for a given visibility set.

    This file contains only prompts (no answers) for students to solve.

    Args:
        output_json_path: Path where the tests JSON will be written
        visibility: Either "public" or "private" to filter exercises
    """
    filtered_exercises = get_exercises_by_visibility(visibility)
    corrections = generate_function_calling_corrections(filtered_exercises)
    # Only include prompts, exclude answers
    corrections_list = [
        correction.model_dump(exclude={"expected_output", "parameters", "name"})
        for correction in corrections
    ]

    with open(output_json_path, "w") as f:
        json.dump(corrections_list, f, indent=2)
