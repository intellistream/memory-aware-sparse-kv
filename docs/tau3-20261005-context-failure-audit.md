# 2026-10-05 τ³ context construction failure

Run `deepseek_20261005T014954Z_1db56e05` failed after collecting 114 eligible
retail and 89 eligible banking episodes. The first pair failed with
`Context target unavailable for retail/11/tool_call/8192`; no pairs, trace phases
or CPU replays were produced. Its failed status and 1870-file hash-verified
archive remain intact. Original service health and final publication were
verified.

The adapter stopped each task at its first eligible tool result, then selected
the shortest episodes for history. Measured with the pinned Pod tokenizer, all
complete fragments in a retail 8K chain held only 2668 or 2702 tokens; the
32K chains held 5794 or 5813. Even all 24 selected retail tasks of one chain
held only 6969 or 7022 tokens. Therefore reordering those fragments could not
meet the 8K/32K coverage gate. Banking history had sufficient aggregate length.

The replacement collector retains the first eligible native event as an anchor
and continues retail through natural simulator termination. Pair selection uses
distinct task histories and exact tokenizer lengths, recording per-cell
feasibility. It fails explicitly if genuine histories still cannot fill a cell.
