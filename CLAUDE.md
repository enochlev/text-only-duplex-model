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

Active reward functions (`trainer.py`) in order, with weights `[2.0, 4.0, 2.0, 2.25, 0.75, 2.0]`:

| # | Function | Weight | Step type | Fires when | Raw values | Weighted values |
|---|---|---|---|---|---|---|
| RM1 | `block_silence_penalty` | 2.0 | **Idle** | Bot stays silent after user finishes speaking (capped at 2 blocks; model is force-idled beyond that) | lag=0: −1.0 / lag=1: −2.0 / lag≥2: 0.0 | −2.0 / −4.0 / 0.0 |
| RM2 | `block_interruption_penalty` | 4.0 | **Speech** | Bot speaks while user is also speaking. First overlap free only if source block T had no user speech. | committed(run=1,silent src): 0.0 / true-interrupt(run=1): −0.75 / run=2: −1.0 / run=3: −1.5 / run≥4: −2.0 | 0.0 / −3.0 / −4.0 / −6.0 / −8.0 |
| RM3 | `block_idle_reward` | 2.0 | **Idle** | Bot stays silent while user is mid-sentence AND user continues in the next block (post-episode lookahead) | +0.5 | +1.0 |
| RM4 | `timely_response_reward` | 2.25 | **Speech** | Bot speaks (non-overlap) promptly after user finishes their turn. **No bonus if the source block already had bot speech** (that's an interruption, not a polite wait — RM2 handles it). | lag=0: +1.0 / lag=1: +0.75 / lag=2: +0.5 | +2.25 / +1.69 / +1.125 |
| RM5 | `backchannel_loop_penalty` | 0.75 | **Speech** | Bot outputs a backchannel-only response. Single backchannel during user's mid-sentence is free. | mid-sentence run=1: 0.0 / post-turn run=1: −0.5 / run N: −0.5N | 0.0 / −0.375 / −0.375N |
| RM6 | `missed_turn_penalty` | 2.0 | **Speech** | Bot speech step follows N unanswered prior user turns. Each skipped turn costs −1.0. Current turn being answered does not count. Uses base history. | N skipped: −N | −2.0 per skipped turn |

`vad_overlap_penalty` (audio overlap via pyannote OSD) is defined but commented out — no-op in text-only simulation; re-enable for real audio.

`junk_output_penalty` (HTML/markdown junk, was RM6) is **commented out** as of 2026-06-06 — the MiniCPM base no longer emits junk tokens, and the regex penalised the model's natural markdown/list formatting. `missed_turn_penalty` is now RM6 (was RM7). The `_JUNK_RE` guard inside `timely_response_reward` still suppresses RM4 for junk blocks even though the standalone RM is off. Re-enable if a future base model regresses to outputting tags/markdown.

**Key interactions:**
- RM1 + RM4 are complementary: RM1 penalises idle steps that delay a response; RM4 rewards the speech step that delivers it.
- RM6 (`missed_turn_penalty`) is the speech-step complement to RM1: it creates a **direct gradient** on the speech step for having skipped prior turns, which RM1 cannot do (RM1 only propagates via returns through idle steps). Responding to Q1 and Q2 earns RM6=0 both times; skipping Q1 and responding only to Q2 earns RM6=−2.0 on the Q2 speech step.
- RM2 and RM6 (`missed_turn_penalty`) both receive **base history** (ends at source block T). `missed_turn_penalty` uses base history so prior covered blocks from the same LLM call don't falsely break the unanswered-turn count.
- RM2 + RM4 are now strictly complementary on interruptions: an interrupt is penalised by RM2 and earns **no** RM4 offset (RM4's source-block guard), so barging in is unprofitable. Previously a barge-in collected +2.5 the moment the user stopped, netting only ≈−0.5.
- RM3 uses **post-episode lookahead** to verify the user truly continued speaking (not just a gap before a new turn).
- RM5 receives **augmented history** so consecutive-backchannel run counts accumulate correctly across all covered blocks of the same step.

---

## 9. Epsilon-Greedy Exploration (`rl_trainer.py:390`)

When the user is mid-sentence, the bot is sometimes **forced silent** even if it would normally generate. This lets REINFORCE observe RM3's +0.5 reward and learn that silence during user speech is correct. The rate is **20%** for clean new-question starts (user has text, source block had no prior bot overlap) and **30%** for overlap moments (source block already had both user and bot text), where staying silent is more clearly correct. (Clean-start rate was raised 10%→20% on 2026-06-06: the verbose MiniCPM base jumps in after the first fragment of a multi-block question, so more forced-idle samples are needed to teach it to wait for the full question.)

**Why `max_tokens=1` for forced-idle steps:** REINFORCE requires a log probability to compute a gradient. Even when the text output is discarded, vLLM generates one token so its log-prob is captured. Without this, `log_probs=[]` and the gradient is zero.

---

## 10. Known Issue — Fully-Idle Episodes

Idle-step rewards (RM1/RM3) propagate via `_compute_returns` onto adjacent speech steps' advantages. If an episode has **zero speech steps**, no gradient is computed at all — the RM1 penalty never reaches the optimizer. Epsilon-greedy exploration and the current RM weights (`[2.0, 4.0, 2.0, 2.25, 0.75, 2.0]`) mitigate this, but if silent episodes dominate, consider forcing at least one speech step per episode (analogous to forced-idle epsilon).

---

## 11. Hyperparameter & RM Change Log

Entries are newest-first. Format: `date | param | old → new | why (5–15 words)`.

| Date | Parameter / File | Old | New | Why |
|---|---|---|---|---|
| 2026-06-06 | RM3 weight (`trainer.py`) | 1.5 | 2.0 | RM4≈−RM2 cancelled to a flat 50-step plateau; tilt the balance toward "wait" |
| 2026-06-06 | RM4 weight (`trainer.py`) | 2.5 | 2.25 | Same plateau fix — slightly reduce speak incentive so net gradient favours caution |
| 2026-06-06 | RM4 source-block guard (`rewards.py`) | +1.0 whenever src had user_text | +1.0 only if src had no bot speech | Barge-in (interrupt) collected the timely +2.5 the moment user stopped, halving RM2's deterrent; now interrupts are strictly unprofitable |
| 2026-06-06 | RM6 `junk_output_penalty` (`trainer.py`) | active, weight=1.5 | commented out | MiniCPM base is clean; RM6 penalised its natural markdown/list formatting. missed_turn renumbered RM7→RM6 |
| 2026-06-06 | Epsilon-greedy clean-start rate (`rl_trainer.py`) | 0.10 | 0.20 | Verbose MiniCPM jumps in after first question fragment; more forced-idle needed to teach waiting through multi-block questions |
| 2026-05-25 | RM2 run=2 raw penalty (`rewards.py`) | −0.5 | −1.0 | First conscious re-interrupt decision was as cheap as committed-overlap free pass |
| 2026-05-25 | RM2 run=3 raw penalty (`rewards.py`) | −1.0 | −1.5 | Proportionate escalation after run=2 change |
| 2026-05-25 | RM2 true-interrupt run=1 raw penalty (`rewards.py`) | −0.5 | −0.75 | Model learned overlap+response (+2.5 RM4) beats first-interrupt cost (-2.0); raising to -3.0 weighted makes it unprofitable |
| 2026-05-25 | RM7 `missed_turn_penalty` added (`rewards.py`) | — | weight=2.0 | Direct speech-step gradient for skipping prior user turns; RM1 propagates too weakly |
| 2026-05-25 | RM1 lag≥2 penalty (`rewards.py`) | −3.0 | 0.0 | Model force-idled beyond max_blocks_after_user_speech=2; perpetual -6.0/block was noise |
| 2026-05-25 | `kl_ref_coeff` (`trainer.py`) | 0.075 | 0.04 | SFT ref model is silent; high KL coeff anchored student to silence, blocking RM4 |
| 2026-05-25 | `vllm_temperature` (`trainer.py`) | 0.8 | 1.0 | Low temp sharpened EOS distribution; 1.0 restores natural sampling variance |
| 2026-05-25 | RM4 weight (`trainer.py`) | 1.5 | 2.5 | Model converged to silence; +2.5 now clearly beats −2.0 interrupt risk |
| 2026-05-25 | RM1 weight (`trainer.py`) | 1.5 | 2.0 | Silence penalty dominant but gradient blocked in fully-silent episodes; heavier weight improves mixed episodes |
| 2026-05-25 | RM4 turn-complete check (`rewards.py`) | `_user_finished_in(src)` | removed | Post-episode: T+1 silent IS the lookahead; punctuation redundant |
| 2026-05-25 | RM5 mid-sentence check (`rewards.py`) | `src.user_text[-1] not in _TERM` | `bool(block.user_text)` | Lookahead replaces punctuation; fixed "Right." false-free-pass |
| 2026-05-25 | `baseline_ema_alpha` (`rl_trainer.py`) | 0.07 | 0.15 | Baseline drifted to −1.2; inflated advantages pushed model to over-speak |
| 2026-05-25 | RM2 weight (`trainer.py`) | 3.0 | 4.0 | Interruption penalty not strong enough relative to timely-response reward |
| 2026-05-25 | RM2 run=2 raw penalty (`rewards.py`) | −0.5 | −1.0 | First conscious re-interrupt decision was as cheap as committed-overlap free pass |
| 2026-05-25 | RM2 run=3 raw penalty (`rewards.py`) | −1.0 | −1.5 | Proportionate escalation after run=2 change |
| 2026-05-25 | Advantage computation (`rl_trainer.py`) | `G - self._baseline` (EMA) | `(G - μ) / (σ + ε)` per-batch z-score | Baseline drifted to −4.0 while avg_reward ~−0.8; inflated advantages were teaching interruptions as "above average" |
| 2026-05-25 | RM6 junk regex (`rewards.py`) | HTML tags only | + markdown: `**`, ` ``` `, `^# `, `^- ` | Model outputting markdown formatting in later steps; not TTS-speakable |
| 2026-05-25 | `_BACKCHANNELS` (`rewards.py`) | — | added `"ai"` | Model using single-token "AI" as a filler response; escaped RM5 and got RM4 reward |

---

## 12. Key Invariants

1. `history[-1]` passed to any reward function is always the **source block T** — the block where the generation decision was made.
2. All covered blocks from one step share the same base history (up to T). They were committed as part of one atomic LLM call.
3. `interruption_penalty` always uses the **base history** (not augmented). This ensures T+2 is not penalised for T+1's user overlap when both were committed before the user spoke.
4. All other RMs use **augmented history** (base + prior covered blocks) so run-length counters (backchannel, etc.) accumulate correctly within a step.
5. A block with `assistant_text_stale=True` is still included in `blocks_covered` and still scored. Staleness affects prompt visibility, not reward attribution.
6. Pending words that were never committed (cleared by `_invalidate_future_assistant_continuation`) never appear in `blocks_covered` — those blocks have empty `assistant_text`.
