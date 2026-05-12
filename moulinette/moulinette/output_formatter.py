# ABOUTME: Provides formatted terminal output for grading results.
# ABOUTME: Uses ASCII art and optional colors for clear visual separation.

import os
import sys

# Try to use colorama if available, otherwise fall back to no colors
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

# Check if we're in a terminal that supports colors
def _supports_color() -> bool:
    if not COLORS_AVAILABLE:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


class ColoredOutput:
    """Utility class for consistent formatted terminal output during grading."""

    def __init__(self):
        self.use_color = _supports_color()

    def _color(self, text: str, fore: str = "", style: str = "") -> str:
        """Apply color if available, otherwise return plain text."""
        if not self.use_color:
            return text
        return f"{fore}{style}{text}{Style.RESET_ALL if COLORS_AVAILABLE else ''}"

    def separator(self) -> None:
        """Print a separator line."""
        line = "-" * 80
        if self.use_color:
            print(Fore.CYAN + line)
        else:
            print(line)

    def success(self, msg: str) -> None:
        """Print a success message."""
        if self.use_color:
            print(f"{Fore.GREEN}{Style.BRIGHT}{msg}")
        else:
            print(f"[OK] {msg}")

    def error(self, msg: str) -> None:
        """Print an error message."""
        if self.use_color:
            print(f"{Fore.RED}{Style.BRIGHT}{msg}")
        else:
            print(f"[ERROR] {msg}")

    def warning(self, msg: str) -> None:
        """Print a warning message."""
        if self.use_color:
            print(f"{Fore.YELLOW}{msg}")
        else:
            print(f"[WARN] {msg}")

    def info(self, msg: str) -> None:
        """Print an info message."""
        if self.use_color:
            print(f"{Fore.WHITE}{msg}")
        else:
            print(msg)

    def expected(self, label: str, value: object) -> None:
        """Print an expected value."""
        if self.use_color:
            print(f"{Fore.CYAN}  {label}: {Fore.WHITE}{value}")
        else:
            print(f"  [EXPECTED] {label}: {value}")

    def actual(self, label: str, value: object) -> None:
        """Print an actual value."""
        if self.use_color:
            print(f"{Fore.MAGENTA}  {label}: {Fore.WHITE}{value}")
        else:
            print(f"  [GOT]      {label}: {value}")

    def prompt(self, prompt_text: str) -> None:
        """Print the prompt being tested."""
        if self.use_color:
            print(f"{Fore.BLUE}Prompt: {Fore.WHITE}{prompt_text}")
        else:
            print(f"Prompt: {prompt_text}")

    def test_header(self, test_num: int, total: int) -> None:
        """Print a test case header with ASCII box."""
        header = f" Test {test_num}/{total} "
        width = 60
        padding = (width - len(header)) // 2

        if self.use_color:
            print(f"\n{Fore.CYAN}+{'-' * width}+")
            print(f"|{' ' * padding}{header}{' ' * (width - padding - len(header))}|")
            print(f"+{'-' * width}+{Style.RESET_ALL}")
        else:
            print(f"\n+{'-' * width}+")
            print(f"|{' ' * padding}{header}{' ' * (width - padding - len(header))}|")
            print(f"+{'-' * width}+")

    def test_result(self, passed: bool, reason: str = "") -> None:
        """Print a test result indicator."""
        if passed:
            if self.use_color:
                print(f"{Fore.GREEN}{Style.BRIGHT}>>> VALID <<<{Style.RESET_ALL}")
            else:
                print(">>> VALID <<<")
        else:
            if self.use_color:
                print(f"{Fore.RED}{Style.BRIGHT}>>> INVALID: {reason} <<<{Style.RESET_ALL}")
            else:
                print(f">>> INVALID: {reason} <<<")

    def summary(self, score: int, total: int) -> None:
        """Print final score summary with ASCII box."""
        pct = (score / total) * 100 if total else 0

        # Determine status
        if pct == 100:
            status = "PERFECT"
            color = Fore.GREEN if self.use_color else ""
        elif pct >= 70:
            status = "PASSED"
            color = Fore.YELLOW if self.use_color else ""
        else:
            status = "FAILED"
            color = Fore.RED if self.use_color else ""

        score_text = f"SCORE: {score}/{total} ({pct:.1f}%)"

        width = 44

        print()
        if self.use_color:
            print(f"{color}{Style.BRIGHT}+{'=' * width}+")
            print(f"|{' ' * width}|")
            print(f"|{status:^{width}}|")
            print(f"|{score_text:^{width}}|")
            print(f"|{' ' * width}|")
            print(f"+{'=' * width}+{Style.RESET_ALL}")
        else:
            print(f"+{'=' * width}+")
            print(f"|{' ' * width}|")
            print(f"|{status:^{width}}|")
            print(f"|{score_text:^{width}}|")
            print(f"|{' ' * width}|")
            print(f"+{'=' * width}+")
        print()
