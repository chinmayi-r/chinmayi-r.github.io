---
layout: post
title: "Running OpenEvolve: is it searching or remembering?"
date: 2026-06-01 12:00:00 -0400
categories: llms evolution
---

OpenEvolve is an open-source take on DeepMind's AlphaEvolve. You hand it a starting
program and a function that scores any program, and it loops: an LLM proposes edits,
the edits get scored, the good ones stay in a population, repeat. The LLM is acting as
the mutation operator that a normal genetic algorithm would do with random tweaks.

I wanted to poke at one question. When OpenEvolve "improves" a program, is the LLM
actually searching for a better algorithm, or is it just pulling something it already
knows out of its training data? This post is a log of how I tried to tell those apart.
Small number of runs, so I wouldn't lean too hard on any single number here.

## The default example

The repo ships with a function minimization task: find the global minimum of
`f(x,y) = sin(x)cos(y) + sin(xy) + (x^2+y^2)/20`. The starting program is plain random
search. I ran it for a while and watched what the population converged to.

Early on, the best program did something I didn't expect: it hardcoded a starting point
near the answer.

```python
best_x, best_y = -1.5, 0.7  # near known global minimum
```

The true minimum sits around (-1.704, 0.678). So the LLM knew roughly where to look and
then bolted some local refinement on top. That reads more like remembering than
searching. By the time I let it run longer, the best program had shifted to a more
general multi-start simulated annealing approach that didn't lean on the hardcoded point,
and the two scored about the same.

## A function it hasn't seen

The obvious problem with the default task is that the LLM has almost certainly seen that
exact function before. So I made one up with arbitrary coefficients:
f(x,y) = sin(3.7x+1.2)cos(2.3y-0.8) + sin(1.1xy+2.7) + 0.3cos(4.1x-1.7y)+ (x^2+y^2)/25 + 0.2sin(x^2-y)

To check it was actually unfamiliar, I just asked the two models to guess the minimum
directly. Neither got close (the real one is around (-1.548, -1.149)). One reasoned its
way to the wrong place, the other guessed coordinates that were off.

Run on this function, OpenEvolve didn't hardcode anything. It built a general optimizer:
grid scan, random sampling, then scipy's L-BFGS-B from the best few points. It got a
clean result, slightly better than what it managed on the function it "knew."

So the behavior split along a line: when the LLM recognized the problem, it shortcut to
a remembered answer; when it didn't, it assembled a general method out of pieces it knew
(grid search, scipy). Both are kinds of retrieval, just at different granularity.

## A combinatorial problem

The two tasks above are both smooth optimization, where scipy can do the heavy lifting.
I wanted something where gradient methods don't apply, so I set up a small traveling
salesman problem: 20 cities, score is total tour length.

This one hit a good score almost immediately. The program it landed on was
nearest-neighbor plus 2-opt, which is the textbook approach. No evolution really
needed; the LLM wrote the standard thing on basically the first try.

Two of the runs did try to go past 2-opt into or-opt (a fancier local move) and got the
implementation wrong both times, crashing the same way. So it recognized that a better
heuristic existed but couldn't write it correctly, and the loop just fell back to the
working 2-opt version.

## What I took away

Across all three tasks, I didn't see OpenEvolve come up with anything that wasn't already
sitting in the model's training distribution. When the model knew the answer it shortcut
to it; when it didn't, it stitched together known building blocks. The evolutionary part
seemed to mostly filter out broken code and pick among attempts, rather than discover
anything new.

That's not really a knock on OpenEvolve. The original AlphaEvolve results were on hard
problems where the model's prior isn't enough on its own, and those are probably where
this kind of loop earns its keep. My tasks were arguably too easy for the model, which
might be exactly why evolution didn't add much. The interesting regime is probably
problems where the LLM has relevant pieces but not the whole answer, so there's actually
something for the loop to recombine.

I'd want more runs before saying any of this with confidence, but the split between
"recognized, so retrieved" and "didn't recognize, so assembled" was consistent enough
across the three tasks that it seemed worth writing down.