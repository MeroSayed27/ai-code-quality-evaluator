"""
generate_dataset.py

Synthetic bug-injection dataset generator, built on mutation testing.

Produces labeled examples of Python functions (correct vs. buggy) by
applying common real-world bug patterns to a set of seed algorithms.
This is the training/evaluation data for the Phase 2 classifier that
scores AI-generated code for correctness -- the same category of task
used by AI-training platforms (Mindrift, Alignerr, Mercor) to evaluate
coding agents like Claude Code, Cursor, and Copilot.

Usage:
    python generate_dataset.py

Output:
    code_quality_dataset.jsonl  (one JSON object per line)
"""

import json
import re
import os
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Seed functions: correct, well-known small algorithms
# ---------------------------------------------------------------------------

SEEDS = {
    "is_palindrome": '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]''',

    "binary_search": '''def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1''',

    "factorial": '''def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)''',

    "find_max": '''def find_max(nums):
    max_val = nums[0]
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val''',

    "count_vowels": '''def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for ch in s:
        if ch in vowels:
            count += 1
    return count''',

    "reverse_string": '''def reverse_string(s):
    return s[::-1]''',

    "fibonacci": '''def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a''',

    "is_prime": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True''',

    "remove_duplicates": '''def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result''',

    "sum_of_digits": '''def sum_of_digits(n):
    n = abs(n)
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total''',

    "bubble_sort": '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',

    "merge_sorted_lists": '''def merge_sorted_lists(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result''',

    "gcd": '''def gcd(a, b):
    while b:
        a, b = b, a % b
    return a''',

    "count_words": '''def count_words(s):
    return len(s.split())''',

    "is_anagram": '''def is_anagram(a, b):
    return sorted(a) == sorted(b)''',

    "linear_search": '''def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1''',
}

# ---------------------------------------------------------------------------
# 2. Mutation operators: common real-world bug patterns.
#    Each takes source code, returns (mutated_code, applied_bool).
# ---------------------------------------------------------------------------

def mutate_off_by_one(code):
    if " - 1" in code:
        return code.replace(" - 1", "", 1), True
    new = re.sub(r"range\((\w+)\)", r"range(\1 - 1)", code, count=1)
    if new != code:
        return new, True
    return code, False


def mutate_comparison_flip(code):
    for op, flipped in [("<=", "<"), (">=", ">"), ("==", "!="), ("<", "<="), (">", ">=")]:
        if op in code:
            return code.replace(op, flipped, 1), True
    return code, False


def mutate_boundary_removal(code):
    m = re.search(r"    if .+?:\n        return .+?\n", code)
    if m:
        return code.replace(m.group(0), "", 1), True
    return code, False


def mutate_wrong_operator(code):
    for op, wrong in [(" + ", " - "), (" and ", " or "), (" - ", " + ")]:
        if op in code:
            return code.replace(op, wrong, 1), True
    return code, False


def mutate_swapped_variables(code):
    if "j]" in code:
        return code.replace("j]", "i]", 1), True
    return code, False


def mutate_increment_removed(code):
    for pat in ["+= 1", "-= 1"]:
        if pat in code:
            idx = code.find(pat)
            return code[:idx] + code[idx + len(pat):], True
    return code, False


MUTATIONS = [
    ("off_by_one", mutate_off_by_one),
    ("comparison_flip", mutate_comparison_flip),
    ("boundary_removal", mutate_boundary_removal),
    ("wrong_operator", mutate_wrong_operator),
    ("swapped_variables", mutate_swapped_variables),
    ("increment_removed", mutate_increment_removed),
]

# ---------------------------------------------------------------------------
# 3. Build the dataset
# ---------------------------------------------------------------------------

def build_dataset():
    rows = []
    row_id = 0

    for name, code in SEEDS.items():
        rows.append({
            "id": row_id,
            "function_name": name,
            "code": code,
            "label": 1,          # 1 = correct
            "bug_type": "none",
        })
        row_id += 1

        seen_variants = {code}
        for bug_type, mutator in MUTATIONS:
            mutated, applied = mutator(code)
            if applied and mutated not in seen_variants:
                seen_variants.add(mutated)
                rows.append({
                    "id": row_id,
                    "function_name": name,
                    "code": mutated,
                    "label": 0,   # 0 = buggy
                    "bug_type": bug_type,
                })
                row_id += 1

    return rows


def main():
    rows = build_dataset()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "code_quality_dataset.jsonl")

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_correct = sum(1 for r in rows if r["label"] == 1)
    n_buggy = sum(1 for r in rows if r["label"] == 0)
    print(f"Generated {len(rows)} examples -> {out_path}")
    print(f"  correct: {n_correct}")
    print(f"  buggy:   {n_buggy}")
    print("Bug type breakdown:")
    for bug_type, c in Counter(r["bug_type"] for r in rows if r["label"] == 0).items():
        print(f"  {bug_type}: {c}")


if __name__ == "__main__":
    main()
