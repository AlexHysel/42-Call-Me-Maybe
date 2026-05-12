# ABOUTME: Extracts function metadata (name, description, params, return type) for JSON export.
# ABOUTME: Used to generate function definitions that students receive as input.

import json
from typing import Dict, List, Callable, get_type_hints

from pydantic import BaseModel

from moulinette.functions_definition import get_functions_by_visibility


# Map Python types to JSON schema types
TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


class ParameterInfo(BaseModel):
    type: str


class FunctionInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo


def extract_function_info(fn: Callable) -> FunctionInfo:
    """Extract metadata from a function object including docstring."""
    # Get function name
    name = fn.__name__

    # Get docstring (first line only, strip whitespace)
    description = ""
    if fn.__doc__:
        description = fn.__doc__.strip().split("\n")[0]

    # Get argument names from signature
    args_names = list(fn.__code__.co_varnames[:fn.__code__.co_argcount])

    # Get type hints including return type
    type_hints = get_type_hints(fn)

    # Get return type
    return_type_raw = type_hints.pop('return', str).__name__
    return_type = TYPE_MAP.get(return_type_raw, "string")

    # Build parameters dict
    parameters = {}
    for arg_name in args_names:
        if arg_name in type_hints:
            type_raw = type_hints[arg_name].__name__
            param_type = TYPE_MAP.get(type_raw, "string")
            parameters[arg_name] = ParameterInfo(type=param_type)

    return FunctionInfo(
        name=name,
        description=description,
        parameters=parameters,
        returns=ParameterInfo(type=return_type)
    )


def generate_function_calling_definition(
    output_json_path: str,
    visibility: str = "public",
) -> None:
    """Generate function definitions JSON file for a given visibility set.

    Args:
        output_json_path: Path where the JSON file will be written
        visibility: Either "public" or "private" to filter functions
    """
    # Get functions filtered by visibility
    functions = get_functions_by_visibility(visibility)

    # Extract info from each function
    function_infos = [extract_function_info(fn) for fn in functions]

    # Convert to dict format for JSON
    function_infos_list = [fn.model_dump() for fn in function_infos]

    with open(output_json_path, "w") as f:
        json.dump(function_infos_list, f, indent=2)
