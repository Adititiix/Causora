# Prompt: write harder evaluation questions (optional)

The seed set (`eval/questions.seed.jsonl`) has 50 questions, mostly one-document lookups.
That is a fair baseline, but real retrieval problems show up on harder ones. Use this prompt to
draft more, then **verify each answer yourself against the files**. An eval set with wrong
ground truth is worse than a small one.

Save the result to `eval/questions.extra.jsonl`. Aim for 15-25 extras so you end above 65 total.

---

## Prompt (copy everything below this line)

I am evaluating a RAG system over a synthetic corpus of data-pipeline runbooks, Airflow logs,
table schema docs, and past incident reports. I will paste a few corpus files below. Write
evaluation questions in JSONL, one object per line, with exactly these fields:

```
{"external_id": "X001", "question": "...", "ground_truth": "...", "category": "...",
 "should_abstain": false, "expected_paths": ["relative/path/one", "relative/path/two"]}
```

Allowed `category` values: log_diagnosis, runbook, schema, incident, multi_hop, unanswerable.

Write these kinds of questions (mix them):

1. Multi_hop (need two documents): e.g. "The transform_orders task failed with a KeyError.
   Which upstream table changed, who owns it, and when?" (needs a log + a schema doc).
2. Paraphrase: ask about a runbook WITHOUT using any word that appears in its title or
   headings (tests whether vector search beats keyword search).
3. Exact-token: ask using only an exact error string or constraint name (tests whether
   keyword search beats vector search).
4. Near-miss traps: a question that sounds like one runbook but belongs to a similar one
   (e.g. SIGKILL out-of-memory vs SIGTERM node reclaim).
5. Unanswerable: plausible questions whose answer is genuinely NOT in the files. For these use
   `"should_abstain": true`, `"ground_truth": "NOT_ANSWERABLE"`, `"expected_paths": []`.

Rules:

- `ground_truth` must be fully supported by the pasted files. Do not use outside knowledge.
- `expected_paths` must list every file needed to answer, using the exact relative paths I give.
- Use ids X001, X002, ... so they never clash with the seed ids (Q001-Q050).
- If you are not certain an answer is supported, drop the question.

Corpus files (path, then contents):

<paste 5-8 files here, each preceded by its path, e.g. `runbooks/oom-killed.md`>
