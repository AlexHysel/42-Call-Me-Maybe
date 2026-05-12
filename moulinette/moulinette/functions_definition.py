# ABOUTME: Defines all functions available for the function calling exercise.
# ABOUTME: Includes visibility metadata (public/private) and test cases for each function.

import math
from typing import Callable, Any


# =============================================================================
# PUBLIC FUNCTIONS - given to students for practice
# =============================================================================

def fn_add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return their sum."""
    assert isinstance(a, float)
    assert isinstance(b, float)
    return a + b


def fn_greet(name: str) -> str:
    """Generate a greeting message for a person by name."""
    assert isinstance(name, str)
    return f"Hello, {name}!"


def fn_reverse_string(s: str) -> str:
    """Reverse a string and return the reversed result."""
    assert isinstance(s, str)
    return s[::-1]


def fn_get_square_root(a: float) -> float:
    """Calculate the square root of a number."""
    assert isinstance(a, float)
    return math.sqrt(a)


def fn_substitute_string_with_regex(source_string: str, regex: str, replacement: str) -> str:
    """Replace all occurrences matching a regex pattern in a string."""
    import re
    assert isinstance(source_string, str)
    assert isinstance(regex, str)
    assert isinstance(replacement, str)
    return re.sub(regex, replacement, source_string)


# =============================================================================
# PRIVATE FUNCTIONS - used for grading, not shown to students
# =============================================================================

def fn_multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return their product."""
    assert isinstance(a, float)
    assert isinstance(b, float)
    return a * b


def fn_is_even(n: int) -> bool:
    """Check if an integer is even, returns True if even, False if odd."""
    assert isinstance(n, int)
    return n % 2 == 0


def fn_calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest: principal * (1 + rate)^years."""
    assert isinstance(principal, float)
    assert isinstance(rate, float)
    assert isinstance(years, int)
    return principal * ((1 + rate) ** years)


def fn_execute_sql_query(query: str, database: str) -> str:
    """Execute a SQL query on a specified database."""
    assert isinstance(query, str)
    assert isinstance(database, str)
    return f"Executing on {database}: {query}"


def fn_read_file(path: str, encoding: str) -> str:
    """Read a file from the given path with specified encoding."""
    assert isinstance(path, str)
    assert isinstance(encoding, str)
    return f"Reading {path} with {encoding}"


def fn_format_template(template: str) -> str:
    """Format a template string containing placeholders."""
    assert isinstance(template, str)
    return f"Template: {template}"


# =============================================================================
# EXERCISES DICTIONARY
# Keys are function objects, values include visibility and test cases
# =============================================================================

exercises: dict[Callable, dict[str, Any]] = {
    # --- PUBLIC FUNCTIONS (11 tests) ---
    fn_add_numbers: {
        "visibility": "public",
        "tests": [
            {"prompt": "What is the sum of 2 and 3?", "fn_args": {"a": 2., "b": 3.}},
            {"prompt": "What is the sum of 265 and 345?", "fn_args": {"a": 265., "b": 345.}},
        ]
    },
    fn_greet: {
        "visibility": "public",
        "tests": [
            {"prompt": "Greet shrek", "fn_args": {"name": "shrek"}},
            {"prompt": "Greet john", "fn_args": {"name": "john"}},
        ]
    },
    fn_reverse_string: {
        "visibility": "public",
        "tests": [
            {"prompt": "Reverse the string 'hello'", "fn_args": {"s": "hello"}},
            {"prompt": "Reverse the string 'world'", "fn_args": {"s": "world"}},
        ]
    },
    fn_get_square_root: {
        "visibility": "public",
        "tests": [
            {"prompt": "What is the square root of 16?", "fn_args": {"a": 16.}},
            {"prompt": "Calculate the square root of 144", "fn_args": {"a": 144.}},
        ]
    },
    fn_substitute_string_with_regex: {
        "visibility": "public",
        "tests": [
            {
                "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
                "fn_args": {
                    "source_string": "Hello 34 I'm 233 years old",
                    "regex": "\\d+",
                    "replacement": "NUMBERS",
                },
            },
            {
                "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
                "fn_args": {
                    "source_string": "Programming is fun",
                    "regex": "[aeiouAEIOU]",
                    "replacement": "*",
                },
            },
            {
                "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
                "fn_args": {
                    "source_string": "The cat sat on the mat with another cat",
                    "regex": "\\bcat\\b",
                    "replacement": "dog",
                },
            },
        ]
    },
    # --- PRIVATE FUNCTIONS (11 tests) ---
    fn_multiply_numbers: {
        "visibility": "private",
        "tests": [
            {"prompt": "What is the product of 3 and 5?", "fn_args": {"a": 3., "b": 5.}},
            {"prompt": "What is the product of 12 and 4?", "fn_args": {"a": 12., "b": 4.}},
        ]
    },
    fn_is_even: {
        "visibility": "private",
        "tests": [
            {"prompt": "Is 4 an even number?", "fn_args": {"n": 4}},
            {"prompt": "Is 7 an even number?", "fn_args": {"n": 7}},
        ]
    },
    fn_calculate_compound_interest: {
        "visibility": "private",
        "tests": [
            {
                "prompt": "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years",
                "fn_args": {"principal": 1234567.89, "rate": 0.0375, "years": 23}
            },
        ]
    },
    fn_execute_sql_query: {
        "visibility": "private",
        "tests": [
            {
                "prompt": "Execute SQL query 'SELECT * FROM users' on the production database",
                "fn_args": {
                    "query": "SELECT * FROM users",
                    "database": "production"
                }
            },
            {
                "prompt": "Run the query 'INSERT INTO logs VALUES (1, 2, 3)' on the system database",
                "fn_args": {
                    "query": "INSERT INTO logs VALUES (1, 2, 3)",
                    "database": "system"
                }
            }
        ]
    },
    fn_read_file: {
        "visibility": "private",
        "tests": [
            {
                "prompt": "Read the file at /home/user/data.json with utf-8 encoding",
                "fn_args": {"path": "/home/user/data.json", "encoding": "utf-8"}
            },
            {
                "prompt": "Read C:\\Users\\john\\config.ini with latin-1 encoding",
                "fn_args": {"path": "C:\\Users\\john\\config.ini", "encoding": "latin-1"}
            },
        ]
    },
    fn_format_template: {
        "visibility": "private",
        "tests": [
            {
                "prompt": "Format template: Hello {user}'s profile!",
                "fn_args": {"template": "Hello {user}'s profile!"}
            },
            {
                "prompt": 'Format template: Say "hello" to {name}',
                "fn_args": {"template": 'Say "hello" to {name}'}
            },
        ]
    }
}


def get_exercises_by_visibility(visibility: str) -> dict[Callable, dict[str, Any]]:
    """Filter exercises by visibility (public or private)."""
    if visibility not in ("public", "private"):
        raise ValueError(f"visibility must be 'public' or 'private', got '{visibility}'")

    return {
        fn: data for fn, data in exercises.items()
        if data["visibility"] == visibility
    }


def get_functions_by_visibility(visibility: str) -> list[Callable]:
    """Get list of function objects by visibility."""
    filtered_exercises = get_exercises_by_visibility(visibility)
    return list(filtered_exercises.keys())
