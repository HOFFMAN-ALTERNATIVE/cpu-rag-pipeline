# Security

## Reporting

Report vulnerabilities privately through GitHub's "Report a vulnerability" button on this repository (Security tab). I aim to reply within 72 hours. Please do not open public issues for exploitable bugs.

## Threat model

- **Assets:** the indexed documents (often internal), API keys for model servers.
- **Actors:** whoever controls the corpus, the index directory, the query text, or the model server URL.
- **Entry points:** corpus files, index files (`index.json`, `vectors.npz`, `chunks.jsonl`, `embedder.json`), CLI flags, responses from the embedding and chat endpoints.
- **Top risks and mitigations:**
  - Code execution from index files: `np.load(..., allow_pickle=False)` and plain JSON, no pickle anywhere.
  - Local file reads through the HTTP clients: base URLs must be `http://` or `https://` with a host; `file://` and other schemes are rejected (tested).
  - Hung model servers: request timeouts (30 s embeddings, 120 s chat).
  - Hallucinated citations: the citation validator drops `[n]` markers that do not match a retrieved source.
  - Secrets: API keys are passed to the client constructors by the caller, never logged, and only sent in the `Authorization` header. Use `https://` for any server that is not on localhost.

## Not provided

- Prompt injection from document contents is not filtered. A document can instruct the model; the answer is only as trustworthy as the corpus.
- The pipeline does not enforce per-user access to documents. Build one index per audience, or filter by metadata before answering.
