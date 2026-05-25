# Full-Duplex RL Training — Architecture Reference

This file explains the system's core concepts for any AI assistant working in this repo. Read it before touching `full_duplex.py`, `trainer/rewards.py`, or `trainer/rl_trainer.py`.

---

## 1. What this project is

A reinforcement-learning training loop that teaches a language model to behave correctly in **full-duplex voice conversations** — i.e., both the user and bot can speak simultaneously, the bot must decide each ~1–2 s "block" whether to speak or stay silent, and the model is punished/rewarded based on turn-taking quality.

---

## 2. The Block Model

### `DuplexAudioBlock` (`full_duplex.py:216`)

A fixed-length time window (~1–2 s). Each block has:

| Field | Meaning |
|---|---|
| `block_id` | Unique ID |
| `user_text` | ASR transcription of what the user said in this window (mutable until frozen) |
| `assistant_text` | What the bot said in this window (committed from pending queue) |
| `assistant_text_stale` | True when ASR revised user text behind this block, making the bot's response outdated; hidden from future prompts but still scored by rewards |
| `response_source_block_id` | The `block_id` of the user block that **triggered** the LLM call whose output ended up here |

---

## 3. The Source-Block → Covered-Block Relationship

**This is the most important concept in the system.**

```
Timeline:  ...  [T-1]  [T]  [T+1]  [T+2]  ...
                        ^      ^      ^
                 source block  covered blocks
```

- **Source block T**: The block at which the LLM was invoked. The bot "observed" T's user speech (or silence) and decided to generate a response. `source_block_id = T.block_id`.
- **Covered blocks T+1, T+2, …**: The blocks that received the committed output words from that generation. Their `response_source_block_id` points back to T.

The bot's **decision** was made at T. The **words** land in T+1, T+2.

### Consequences for reward scoring

- `_prior_history(covered_ids)` = `episode.blocks[:T+1_index]` — history ends at T (inclusive).
- All covered blocks for the same step share the **same** prior history (up to T).
- `history[-1]` is always the source block T.
- If the user starts speaking in T+1 after the bot committed, the bot had **no causal visibility** — it decided at T when the user was silent.
- If the user was **already** speaking in T (`history[-1].user_text` is set) and the bot generated anyway, that is a **true interruption** — the bot had full visibility.

---

## 4. Pending Word Queue

The LLM generates a full response in one call, but words are **distributed across blocks over time**:

1. LLM output → `_pending_words` (via `_update_pending_queue`, `full_duplex.py:941`)
2. Each call to `poll()` commits some words to `_current_block.assistant_text` via `_commit_block_words` (`full_duplex.py:910`)
3. Words are distributed evenly: `n_blocks = ceil(total_words / words_per_block)`; first block gets any remainder
4. Remaining uncommitted words stay in `_pending_words` for future blocks

**If the user speaks again before words are committed:** `_invalidate_future_assistant_continuation` (`full_duplex.py:475`) clears `_pending_words` — those future blocks get no text and are never included in `blocks_covered`.

**If words were already committed** (written to `assistant_text`) before the user spoke: `_mark_assistant_history_stale_from` sets `assistant_text_stale = True` on those blocks. The committed text remains in `assistant_text` (and is still scored by rewards), but is hidden from future prompt context.

---

## 5. ASR Invalidation & `context_version`

When ASR revises the user's transcription:
1. `_invalidate_future_assistant_continuation()` — clears pending (uncommitted) words
2. `_mark_assistant_history_stale_from(earliest_changed_index)` — marks older committed bot text as stale
3. `context_version += 1` — signals that any in-flight LLM calls are now stale

In-flight LLM calls check `context_version` against the version captured at call start (`gen_ctx_ver`). If mismatched, the response is discarded (`full_duplex.py:1078`). This prevents stale history from producing phantom responses.

---

## 6. StepRecord Lifecycle (`trainer/rl_trainer.py:75`)

```
During episode:
  LLM call → StepRecord(source_block_id, response_token_ids, log_probs, is_idle, ...)

Post-episode:
  _fill_blocks_covered()  → fills step.blocks_covered from block.response_source_block_id
  _merge_silent_runs()    → merges consecutive speech steps where user was silent between calls
                            (multiple calls = one "continue speaking" decision)

Reward computation:
  compute_rewards()       → fills step.reward and step.reward_breakdown
```

**`_fill_blocks_covered` (`rl_trainer.py:807`):** Groups blocks by `response_source_block_id`. Only includes blocks with non-empty `assistant_text`. Does NOT filter out stale blocks — stale text was actually spoken and deserves scoring.

**`_merge_silent_runs` (`rl_trainer.py:855`):** When the user is silent across multiple LLM calls (`user_spoke_before=False`), those calls are collapsed into one step. This gives a single advantage signal over the whole segment, reducing gradient variance.

---

## 7. Reward Computation

### Speech steps (`compute_rewards`, `rl_trainer.py:1218`)

```python
history = episode.blocks[:first_covered_block_index]   # same for all covered blocks
for blk_pos, block in enumerate(covered):
    aug_history = history + covered[:blk_pos]           # prior covered blocks appended
    for fn in reward_fns:
        h = history if fn == interruption_penalty else aug_history
        score = fn(block, h, is_terminal)
```

Two history variants are passed:
- **`interruption_penalty`** gets `history` (original, ends at T). If it got `aug_history`, T+2's run counter would inflate because T+1 overlapped — but T+1 was committed *before* the user spoke, so T+2 should not be penalised for it.
- **All other RMs** get `aug_history` (history + prior covered blocks). This ensures, e.g., `backchannel_loop_penalty` sees consecutive backchannel blocks from the same step and counts the run correctly.

### Idle steps (`_idle_rm1_reward`, `rl_trainer.py:1238`)

Idle steps produce no tokens, so REINFORCE can't compute a gradient directly. Their reward is set and propagated back through `_compute_returns` to reduce the advantage of preceding speech steps. History = `episode.blocks[:source_block_index + 1]` (up to and including the source block).

---

## 8. Reward Function Reference

Active reward functions (`trainer.py`) in order, with weights `[1.5, 3.0, 1.5, 1.5, 0.75, 1.5]`:

| # | Function | Weight | Step type | Fires when | Raw values | Weighted values |
|---|---|---|---|---|---|---|
| RM1 | `block_silence_penalty` | 1.5 | **Idle** | Bot stays silent after user finishes speaking | lag=0: −1.0 / lag=1: −2.0 / lag≥2: −3.0 | −1.5 / −3.0 / −4.5 |
| RM2 | `block_interruption_penalty` | 3.0 | **Speech** | Bot speaks while user is also speaking. First overlap free only if source block T had no user speech. | run≤2: −0.5 / run=3: −1.0 / run≥4: −2.0 | −1.5 / −3.0 / −6.0 |
| RM3 | `block_idle_reward` | 1.5 | **Idle** | Bot stays silent while user is mid-sentence AND user continues in the next block (post-episode lookahead) | +0.5 | +0.75 |
| RM4 | `timely_response_reward` | 1.5 | **Speech** | Bot speaks (non-overlap) promptly after user finishes their turn | lag=0: +1.0 / lag=1: +0.75 / lag=2: +0.5 | +1.5 / +1.125 / +0.75 |
| RM5 | `backchannel_loop_penalty` | 0.75 | **Speech** | Bot outputs a backchannel-only response. Single backchannel during user's mid-sentence is free. | mid-sentence run=1: 0.0 / post-turn run=1: −0.5 / run N: −0.5N | 0.0 / −0.375 / −0.375N |
| RM6 | `junk_output_penalty` | 1.5 | **Speech** | Bot outputs HTML/junk tokens (`<idle>`, `<span>`, etc.) instead of speech text | −1.0 | −1.5 |

`vad_overlap_penalty` (audio overlap via pyannote OSD) is defined but commented out — no-op in text-only simulation; re-enable for real audio.

**Key interactions:**
- RM1 + RM4 are complementary: RM1 penalises idle steps that delay a response; RM4 rewards the speech step that delivers it.
- RM2 always receives **base history** (ends at source block T). This prevents T+2 from being penalised for T+1's overlap when both were committed before the user spoke.
- RM3 uses **post-episode lookahead** to verify the user truly continued speaking (not just a gap before a new turn).
- RM5 receives **augmented history** so consecutive-backchannel run counts accumulate correctly across all covered blocks of the same step.

---

## 9. Epsilon-Greedy Exploration (`rl_trainer.py:390`)

10 % of the time, when the user is mid-sentence, the bot is **forced silent** even if it would normally generate. This lets REINFORCE observe RM5's +0.5 reward and learn that silence during user speech is correct.

**Why `max_tokens=1` for forced-idle steps:** REINFORCE requires a log probability to compute a gradient. Even when the text output is discarded, vLLM generates one token so its log-prob is captured. Without this, `log_probs=[]` and the gradient is zero.

---

## 10. Known Issue — Fully-Idle Episodes

Idle-step rewards (RM1/RM3) propagate via `_compute_returns` onto adjacent speech steps' advantages. If an episode has **zero speech steps**, no gradient is computed at all — the RM1 penalty never reaches the optimizer. Epsilon-greedy exploration and the current RM weights (`[1.5, 3.0, 1.5, 1.5, 0.75, 1.5]`) mitigate this, but if silent episodes dominate, consider forcing at least one speech step per episode (analogous to forced-idle epsilon).

---

## 11. Key Invariants

1. `history[-1]` passed to any reward function is always the **source block T** — the block where the generation decision was made.
2. All covered blocks from one step share the same base history (up to T). They were committed as part of one atomic LLM call.
3. `interruption_penalty` always uses the **base history** (not augmented). This ensures T+2 is not penalised for T+1's user overlap when both were committed before the user spoke.
4. All other RMs use **augmented history** (base + prior covered blocks) so run-length counters (backchannel, etc.) accumulate correctly within a step.
5. A block with `assistant_text_stale=True` is still included in `blocks_covered` and still scored. Staleness affects prompt visibility, not reward attribution.
6. Pending words that were never committed (cleared by `_invalidate_future_assistant_continuation`) never appear in `blocks_covered` — those blocks have empty `assistant_text`.
