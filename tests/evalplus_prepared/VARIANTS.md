Prepared EvalPlus rows for `verify-reference`. Apache-2.0.

The full prepared HumanEval+ and MBPP+ suites are about 16 MB (the plus tests alone are 10.9 MB and 4.7 MB). This directory keeps one row for every distinct helper and assertion variant instead of those full files.

Assertion variants upstream:

- MBPP+ standard numeric helper. The body calls `is_floats(exp)` and `np.allclose`. 368 tasks. Row 590.
- MBPP+ set helper. The same numeric helper plus `out = set(out)`. Tasks 2, 7, 111, 140, 232, 579, and 769. Row 232.
- MBPP+ match helper. A bool uses `==`. Anything else uses `exp == (out is not None)`. The body sets `exact_match` and does not assert it. Tasks 737, 787, and 794. All three rows are here.
- HumanEval+ standard helper. The body sets `exact_match`, calls `is_floats(exp)` and `np.testing.assert_allclose`, then asserts `exact_match`. 163 tasks. Rows HumanEval/14, 15, 83, 100, 130, and 139.
- HumanEval/32 has no `assertion` helper.

Other helpers, each kept:

- `is_floats`. One body, on every numeric row. Scoring removes it.
- `ref_func`. HumanEval/14, 15, 83, 100, 130, 139, and MBPP+ 88, 255, and 630.
- `adjac`. MBPP+ 630. Fetch renames `adjac(` to `candidate(` because that function is the first `def` in the reference solution.
- `_poly`. HumanEval/32.

Exclusions in this directory stay exclusions: HumanEval/32, MBPP+ 255, and MBPP+ 596. `verify-reference` requires every other row here to pass.
