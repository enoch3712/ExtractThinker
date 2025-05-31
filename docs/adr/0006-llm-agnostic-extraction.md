# 6. LLM-agnostic extraction

Date: 2024-06-09

## Status

Accepted

## Context

The project aims to support multiple LLMs (Language Model APIs) for extraction, not just a single provider.

## Decision

The extractor will be designed to load and use any LLM backend, with the LLM set at runtime via configuration.

## Consequences

- The system is flexible and can adapt to new LLMs as they become available.
- Users can select the LLM that best fits their needs and budget.