/* One explicit experiment at a time. Incomplete or superseded runs never become observations. */
'use strict';
class PromptExperiment {
  constructor({ request, onProgress, onComplete, onError, wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms)) }) {
    Object.assign(this, { request, onProgress, onComplete, onError, wait });
    this.generation = 0;
    this.job = null;
    this.busy = false;
  }
  async cancel() {
    this.generation += 1;
    this.busy = false;
    const job = this.job;
    this.job = null;
    if (job) { try { await this.request({ action: 'cancel', job_id: job }); } catch (_) {} }
  }
  async start(prompt, checkpoints) {
    // Do not normalize the prompt: whitespace is part of the tokenized experiment.
    if (typeof prompt !== 'string' || !prompt.trim() || [...prompt].length > 400) {
      this.onError('Enter a prompt of 1–400 characters.');
      return;
    }
    const previousCancellation = this.cancel();
    // Claim this generation synchronously. A cancel in the next microtask must
    // not be undone when the previous run's cancellation finishes.
    const generation = this.generation;
    this.busy = true;
    let job;
    const current = () => generation === this.generation;
    this.onProgress({ status: 'starting', completed: 0, total: checkpoints.length });
    try {
      await previousCancellation;
      if (!current()) return;
      const started = await this.request({ action: 'start', prompt });
      job = started.job_id;
      if (typeof job !== 'string' || started.total !== checkpoints.length) throw new Error('The experiment server returned an incompatible run.');
      // A cancel can race with admission. Cancel the accepted job as soon as its ID arrives.
      if (!current()) { await this.request({ action: 'cancel', job_id: job }).catch(() => {}); return; }
      this.job = job;
      const frames = [];
      let metadata, provenance;
      let manifestVerified = false;
      let failures = 0;
      const deadline = Date.now() + 240000;
      while (frames.length < checkpoints.length) {
        if (!current()) return;
        if (Date.now() > deadline) throw new Error('This run took too long. Please try again in a moment.');
        let result;
        try {
          result = await this.request({ action: 'poll', job_id: job, cursor: frames.length });
          failures = 0;
        } catch (error) {
          if (!current()) return;
          if (++failures > 3 || error.permanent) throw error;
          await this.wait(1000 * failures);
          continue;
        }
        if (!current()) return;
        if (result.status === 'error' || result.status === 'cancelled') throw new Error(result.error || 'The experiment was cancelled.');
        if (result.job_id !== job || result.total !== checkpoints.length) throw new Error('The experiment response did not match this run.');
        if (!['queued', 'running', 'complete'].includes(result.status)) throw new Error('The experiment returned an unknown state.');
        if (result.status === 'complete' && !result.frame) throw new Error('The completed experiment is missing a checkpoint. Please try again.');
        if (result.scene || result.provenance || result.checkpoints || result.model || result.frame) {
          // A matching training step alone is insufficient: each observation
          // must come from the exact immutable weights used by this timeline.
          const sameManifest = Array.isArray(result.checkpoints)
            && result.checkpoints.length === checkpoints.length
            && result.checkpoints.every((checkpoint, index) => checkpoint
              && checkpoint.step === checkpoints[index].step
              && /^[a-f0-9]{40}$/i.test(checkpoints[index].commit_sha || '')
              && checkpoint.commit_sha === checkpoints[index].commit_sha);
          if (!sameManifest || result.provenance?.tokenizer_commit_sha !== checkpoints.at(-1).commit_sha) {
            throw new Error('This experiment used a different checkpoint or tokenizer. Please refresh and try again.');
          }
          manifestVerified = true;
        }
        if (result.scene) metadata = result.scene;
        if (result.provenance) provenance = result.provenance;
        this.onProgress(result);
        if (result.frame) {
          if (result.index !== frames.length || result.frame.step !== checkpoints[frames.length].step) throw new Error('A checkpoint arrived out of order. Please try again.');
          frames.push(result.frame);
        } else await this.wait(750);
      }
      if (!metadata || metadata.prompt !== prompt || !provenance || !manifestVerified) throw new Error('The completed experiment is missing its observation record.');
      if (current()) this.onComplete({ ...metadata, custom: true, provenance, frames });
    } catch (error) {
      if (current()) this.onError(error.message || 'The experiment could not be completed.');
      if (job) await this.request({ action: 'cancel', job_id: job }).catch(() => {});
    } finally {
      if (current()) { this.busy = false; this.job = null; }
    }
  }
}
if (typeof module !== 'undefined') module.exports = { PromptExperiment };
