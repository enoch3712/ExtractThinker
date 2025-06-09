# 2. Use a Python monorepo structure

Date: 2024-06-09

## Status

Accepted

## Context

The project consists of multiple components: core logic, document loaders, models, tests, and examples. Managing these as a monorepo simplifies dependency management, testing, and deployment.

## Decision

We will use a single Python repository with the following structure:

- `extract_thinker/` for core code and modules
- `tests/` for all test code
- `examples/` for usage examples and scripts
- `docs/` for documentation
- Project configuration files at the root

## Consequences

- Easier to manage dependencies and code sharing between modules.
- Simplifies CI/CD and testing.
- All code and documentation are versioned together.