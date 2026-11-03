import { JobRecord } from "./jobsApi";

// Imported models downloaded from the same training run are registered as
// separate jobs named "<run> (step N)". This helper collapses them into one
// group per run so the Deploy/Eval pickers (and the model cards) can offer a
// single run entry with a checkpoint dropdown that switches between steps.
const STEP_RE = /^(.*) \(step (\d+)\)$/;

export interface ModelGroup {
  base: string;
  members: JobRecord[]; // sorted by step, latest first
}

// One entry per selectable checkpoint: the real step (for the label) and the
// underlying job id that owns it.
export interface ModelVariant {
  step: number;
  jobId: string;
}

export const modelStepOf = (name: string): number => {
  const m = STEP_RE.exec(name);
  return m ? Number(m[2]) : 0;
};

export const modelBaseName = (name: string): string => {
  const m = STEP_RE.exec(name);
  return m ? m[1] : name;
};

export function groupImportedByRun(jobs: JobRecord[]): ModelGroup[] {
  const order: ModelGroup[] = [];
  const byBase = new Map<string, ModelGroup>();
  for (const job of jobs) {
    const stepped = STEP_RE.test(job.name);
    const base = modelBaseName(job.name);
    if (!stepped) {
      order.push({ base, members: [job] });
      continue;
    }
    let g = byBase.get(base);
    if (!g) {
      g = { base, members: [] };
      byBase.set(base, g);
      order.push(g);
    }
    g.members.push(job);
  }
  for (const g of order) {
    if (g.members.length > 1) {
      g.members.sort((a, b) => modelStepOf(b.name) - modelStepOf(a.name));
    }
  }
  return order;
}

export function variantsOf(g: ModelGroup): ModelVariant[] {
  return g.members.map((m) => ({ step: modelStepOf(m.name), jobId: m.id }));
}
