---
layout: post
title: "Reading nano-vllm, and what happens when it runs out of memory"
date: 2026-06-14 12:00:00 -0400
categories: llms inference systems
---

nano-vllm is a from-scratch reimplementation of vLLM, the LLM inference engine. The "v"
in vLLM is for virtual, as in virtual memory: the KV cache (the stored representations of
past tokens during generation) gets managed the way an OS manages RAM, split into
fixed-size blocks and mapped from logical to physical positions through a page table.

The appeal is that it's small enough to read all the way through. vLLM itself is enormous;
this is the same core ideas in something you can actually follow. I read the whole thing,
then went deep on one question: what happens when the cache fills up mid-generation?

This is a log of what I found, with a small experiment at the end. Take the numbers as
rough.

## How a request moves through

You call `generate()` with prompts. Each gets tokenized, wrapped in a `Sequence`, and put
on a waiting queue. The engine loops, and each step the scheduler picks one of two things
to do: prefill or decode.

Prefill processes a whole prompt at once. Decode generates one token per running request.
The scheduler always clears pending prefills first:

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

which means whenever a request needs prefill, every running request sits idle that step.
That detail matters later.

## The paging system

The cache is split into blocks of 256 tokens. Each request gets a `block_table` mapping
its token positions to physical block IDs:

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
```

Free blocks live in a deque. Allocation pops from the front, freeing appends to the back,
so the oldest freed block gets reused first. The `ref_count` lets requests that share a
prefix point at the same block.

Getting from a block table to an actual memory write goes through a `slot_mapping`, which
a small Triton kernel uses to store the K/V:

```python
slot = tl.load(slot_mapping_ptr + idx)
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
```

So it's three levels: block table, then slot mapping, then physical write. That's the same
shape as an OS going page table to frame number to physical address, which is where the
name comes from.

## Preemption

I was curious what happens when the cache fills up while requests are still generating. I
hadn't really thought about it before and wanted to dig in.

It turns out the trigger isn't "too many requests submitted." The scheduler admits a
request based on how many blocks it needs right now, not how many it'll need at full
length. A 600-token prompt needs 3 blocks at admission, so the scheduler lets it in if 3
are free. But the request grows as it generates, a new block every 256 tokens, and at some
point there are no free blocks left and something has to give:

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
```

`self.running.pop()` grabs the most recently added request, so the policy is LIFO. And
preemption here isn't gentle:

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

The victim's blocks are freed and it goes back to waiting set to re-prefill, so it
recomputes its whole KV cache from scratch later. An OS would pause a process and save its
state; here there's nowhere to save the cache, so the work just gets thrown away and redone.

## The experiment

I set the cache to 136 blocks, sent batches of 4 to 96 requests (600-token prompt, 600
generated each), and logged every preemption.

| N | preempt | wall (s) | tok/s |
|---|---------|----------|-------|
| 4 | 0 | 17.5 | 137 |
| 8 | 0 | 14.0 | 342 |
| 16 | 0 | 14.0 | 687 |
| 32 | 5 | 18.3 | 1050 |
| 64 | 26 | 28.8 | 1332 |
| 96 | 35 | 42.7 | 1349 |

Preemption kicks in at N=32, which matches the rough block math (136 blocks over ~5 blocks
per full request is about 27 concurrent before it gets tight).

I expected throughput to drop once preemption started. It didn't. It rose and flattened
out. What actually moved was wall-clock time, which roughly tripled from N=16 to N=96. The
re-prefill work is parallel and cheap per token, so it barely shows up in tokens-per-second
and instead shows up as latency. If you only watched throughput, preemption would look
free, which it isn't.

One clean thing in the data: the number of prefill calls came out to N plus the number of
preemptions every time, so each eviction costs exactly one extra prefill, no cascade.

## Can the leftover blocks be recovered?

I wondered whether an evicted request could grab its old blocks back through prefix caching
when it restarts, since freeing a block doesn't clear its hash. It mostly can't. At N=96,
35 preemptions freed over 100 blocks and only 4 came back, each recovery a single block.

The reason is that the freed blocks get reused right away by other requests, which clears
their hashes, and that reuse is happening because memory is tight, which is the same thing
that caused the eviction. Freeing happens in reverse order, so a request's first block ends
up reused last and is the only one likely to survive. The chained hash lookup then stops at
the next block, which is already gone. So recovery tops out around one block. Prefix caching
was built for sharing prefixes across different requests, not for rescuing evicted ones, and
the numbers line up with that.

## Trying to do better

Since preemption costs something, I tried three changes to beat the default.

Reserving worst-case blocks at admission so nothing ever gets evicted: this removed
preemption completely but was slower at every N, by about 28% at N=96. Holding space for
full-length requests means fewer run at once and the GPU sits partly idle. The simple fix
turned out worse than the thing it was fixing.

Evicting the cheapest request to redo instead of the newest: about 2% better on a uniform
workload. LIFO already tends to pick a recent, low-token request, so the two end up close.

A mixed workload of short and long requests under both policies: they came out basically
the same. Only long requests got evicted either way, because the short ones finish before
the squeeze happens. The workload decided who got evicted more than the policy did.

## What I took away

nano-vllm goes with optimistic admission and throw-away-and-restart preemption as a
backstop. It's simple, every request finishes, and it degrades into a plateau rather than
falling over. The cost is real but lands on latency instead of throughput, so it's easy to
miss.

None of my three changes clearly beat it. The thing that would actually help is the one
nano-vllm leaves out: swapping evicted cache to CPU memory instead of discarding it, so the
work pauses instead of restarting. That's what real vLLM does, and it's a fair thing to
skip in a codebase this size.

A couple of smaller things I liked while reading: decode replays pre-recorded CUDA graphs
to skip Python overhead (prefill can't, since its lengths vary), and the sampler uses the
Gumbel-max trick instead of `torch.multinomial` to get the same draw in one GPU op.