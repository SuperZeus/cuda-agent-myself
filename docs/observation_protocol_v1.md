# Observation Protocol V1

Observation assembly order:

1. Current task objective
2. Editable files
3. Last compile failure summary
4. Last verify failure summary
5. Last profile summary
6. Current best result
7. Recent diff summaries

Compression rules:

- Keep only the most recent high-value events.
- Truncate stderr to the most relevant tail.
- Deduplicate repeated error blocks.
- Preserve exact file paths and line hints when available.
- Summarize older history into compact bullet-like facts.

Phase-0 implementation:

- The runner stores raw logs.
- Summary generation is stubbed as deterministic truncation and tail extraction.
- Later phases can replace this with model-assisted summarization.
