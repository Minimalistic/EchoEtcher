import sys
import os

# Mock or import the logic
def sanitize_path(path: str) -> str:
    if not path:
        return None
    s = path.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

test_cases = [
    ("/Users/test", "/Users/test"),
    (" /Users/test ", "/Users/test"),
    ('"/Users/test"', "/Users/test"),
    ("'/Users/test'", "/Users/test"),
    (' "/Users/test" ', "/Users/test"),
    ('"/Users/test Space"', "/Users/test Space"),
    ("'/Users/test Space'", "/Users/test Space"),
    ("", None),
    (None, None),
]

failed = 0
for input_path, expected in test_cases:
    result = sanitize_path(input_path)
    if result != expected:
        print(f"FAILED: input={repr(input_path)}, expected={repr(expected)}, got={repr(result)}")
        failed += 1
    else:
        print(f"PASSED: input={repr(input_path)} -> {repr(result)}")

if failed == 0:
    print("\nAll tests passed!")
    sys.exit(0)
else:
    print(f"\n{failed} tests failed.")
    sys.exit(1)
