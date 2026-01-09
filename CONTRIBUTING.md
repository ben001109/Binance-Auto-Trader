# Contributing to Binance Auto Trader (BAT)

To ensure the stability and reliability of the **clean-repo** branch, please follow these rules.

## 🛡️ Protection Rules

1.  **No Direct Pushes**: Never push directly to `clean-repo` or `main`. Always use a Feature Branch.
2.  **Pull Requests**: All changes must be submitted via Pull Request (PR).
3.  **Code Review**: At least one review is required from a Code Owner (`CODEOWNERS`) before merging.
4.  **CI Checks**: The "Flash CI" workflow must pass (Green Build) before merging.

## 🚀 Workflow

1.  **Branching**:
    *   `feat/feature-name` for new features.
    *   `fix/bug-description` for bug fixes.
    *   `chore/maintenance` for docs, CI, or housekeeping.

2.  **Development**:
    *   Install dependencies: `uv sync`
    *   Run integrity checks: `uv run verify_enhancements.py`

3.  **Submission**:
    *   Push your branch.
    *   Open a PR against `clean-repo`.
    *   Ensure CI passes.

## 📝 Style Guide

*   Use `uv` for package management.
*   Update `CHANGELOG.md` for notable changes.
*   Keep `TODO.md` up to date.
