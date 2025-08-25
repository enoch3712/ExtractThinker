# 5. Use Ruff and Flake8 for linting

Date: 2024-06-09

## Status

Accepted

## Context

Consistent code style and static analysis are important for maintainability and code quality.

## Decision

We will use [Ruff](https://github.com/astral-sh/ruff) as the primary linter, with Flake8 as a secondary tool for compatibility and additional checks.

## Consequences

- Code style is enforced automatically.
- Linting errors are caught early in development and CI.
- Contributors must ensure code passes linting before merging.