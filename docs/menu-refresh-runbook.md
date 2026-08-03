# Weekly Cambridge menu refresh

This runbook supplements `scripts/update_menu.py` and the Codex automation. The live Sebastian's Alewife page and the latest GitHub `main` branch are always the sources of truth.

## Source and extraction

1. Resolve the direct PDF, JPEG, or PNG currently linked from the canonical Alewife menu page. Never reuse a prior weekly URL without rediscovering the live link first.
2. Confirm the visible heading is for the current `America/New_York` week and that Monday through Friday are complete.
3. Stop without changes when the source is stale, incomplete, unavailable, or unchanged from `sourceHash`.
4. Perform one Codex multimodal extraction for the whole source. Do not call Gemini or another external LLM during the scheduled refresh.

## Images

1. Select every daily and weekly highlight in the same weekly run.
2. Resolve the item archive slug before generation. Reuse approved archive images and the fixed Taco Tuesday reference; generate only genuine cache misses.
3. Use the ordinary office-cafeteria style in `FOOD_PHOTO_PROMPT_TEMPLATE`: realistic but modest portions, slightly uneven staff plating, practical lighting, and no fine-dining or advertising polish.
4. Inspect each new image before approval. Reject unrelated dishes, illustrations, text or logos, malformed packaging, unsafe-looking food, dimensions below 1024 pixels on an intended axis, and suspiciously small files.

## Binary-safe publication

Large image bytes must not pass through terminal output, captured shell base64, or another text channel that can truncate output.

1. Build a local manifest for every changed binary containing repository path, byte length, SHA-256, Git blob SHA-1, format, and decoded dimensions.
2. Upload each unique image directly from its complete bytes. Reuse the resulting blob for identical daily, weekly, and archive copies rather than uploading the same content repeatedly.
3. Before creating the commit, compare every uploaded blob's Git SHA-1 and byte length with the local manifest. A connector-returned SHA is not sufficient unless it matches the independently calculated local Git blob SHA-1.
4. Independently read back each unique remote blob or raw file, verify its byte length and SHA-256 against the local manifest, decode it as an image, and confirm its dimensions. Abort publication on any mismatch.
5. Build the tree only after all binary checks pass. Recheck the current `main` head immediately before updating the ref, then use a non-forced fast-forward.

## Deployment verification

1. Wait for the GitHub Pages workflow for the exact commit to finish successfully.
2. Read back `menu.json` from the deployed site and confirm the expected source hash and highlight paths.
3. Check every image referenced by the current daily and weekly highlights with a cache-busting query string. Each must return the expected full byte length, decode successfully, and report non-zero dimensions.
4. Open the live dashboard with a cache-busting query and verify that every visible menu card uses its intended generated image. Treat any placeholder URL, incomplete image, zero natural dimensions, or unexpected `currentSrc` as a failed deployment.
5. Report success only after both GitHub remote readback and live Pages verification pass. Browser caching can explain a stale client after a verified deployment, but it must not be used to explain a failed cache-busted check.

## Minimum checks

- `python -m py_compile scripts/update_menu.py`
- Valid JSON and complete Monday-Friday structures
- Referenced file existence and image decoding
- Expected versus unexpected duplicate hashes
- Local-to-remote binary manifest equality
- Latest `main` equals the published commit
- GitHub Pages succeeded and cache-busted live images render without fallbacks
