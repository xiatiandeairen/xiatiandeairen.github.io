// core/checkpoint.ts — In-memory serialization of a model's weights.
//
// WHY this exists: PEFT only makes sense relative to a FIXED pretrained base. stage01
//   trains a toy base on task A; every later stage must start from THAT SAME base so their
//   results are comparable ("LoRA on the stage01 base reaches loss X"). We cannot retrain
//   the base in each stage (drift + wasted time) and we are offline (no disk contract in the
//   book), so we snapshot the base into a plain object and reload it.
//
// THE "FROZEN BASE = SHARED READ-ONLY WEIGHTS" IDEA: a checkpoint is just the flat float
//   buffers keyed by param name. loadBase() copies them into a model's params and (by
//   default) freezes them — exactly the PEFT precondition. Because the dump is deterministic
//   given the seed, any stage that re-trains the base with the same seed gets a bit-identical
//   checkpoint; the book uses that to let each chapter be runnable standalone.
//
// INVARIANT: param identity is by ORDER of Module.parameters(). The loaded model MUST have
//   the same architecture (same params, same order, same shapes) as the dumped one, or
//   loadBase throws. This catches "I changed the model but reloaded an old base" bugs loudly.
//
// FAILURE MODE guarded: silently loading a mismatched checkpoint (wrong shape) would corrupt
//   the base and produce a plausible-but-wrong loss curve. We assert shapes on load.

import { Tensor } from "./tensor.js";
import type { Module } from "./nn.js";

export interface Checkpoint {
  shapes: number[][]; // per-param shape, in Module.parameters() order
  buffers: Float64Array[]; // per-param data copy
}

/** Snapshot a module's current weights (deep copy, so later training cannot mutate it). */
export function dump(model: Module): Checkpoint {
  const params = model.parameters();
  return {
    shapes: params.map((p) => p.shape.slice()),
    buffers: params.map((p) => p.data.slice()),
  };
}

function shapesMatch(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/**
 * Copy checkpoint weights into a model. By default also FREEZES them (PEFT base precondition).
 * @param freeze set false if a stage wants to fully fine-tune from the base (stage02).
 */
export function loadBase(model: Module, ckpt: Checkpoint, freeze = true): void {
  const params = model.parameters();
  if (params.length !== ckpt.buffers.length) {
    throw new Error(`loadBase: model has ${params.length} params but checkpoint has ${ckpt.buffers.length}`);
  }
  params.forEach((p, i) => {
    if (!shapesMatch(p.shape, ckpt.shapes[i])) {
      throw new Error(`loadBase: shape mismatch at param ${i}: model [${p.shape}] vs ckpt [${ckpt.shapes[i]}]`);
    }
    p.data.set(ckpt.buffers[i]);
    p.requires_grad = !freeze; // freeze => requires_grad=false (the base is read-only)
  });
}

/** Freeze a module in place without loading (convenience mirror of Module.freeze). */
export function freeze(model: Module): void {
  for (const p of model.parameters()) p.requires_grad = false;
}

/** Total float slots a checkpoint holds (for honest size reporting). */
export function checkpointFloats(ckpt: Checkpoint): number {
  return ckpt.buffers.reduce((acc: number, b: Float64Array) => acc + b.length, 0);
}

/** Re-export Tensor type usage marker so consumers see the dependency clearly. */
export type { Tensor };
