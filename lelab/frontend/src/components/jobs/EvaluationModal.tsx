import React, { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { JobRecord } from "@/lib/jobsApi";
import EvaluationCard from "@/components/jobs/EvaluationCard";
import { modelStepOf, modelBaseName, ModelVariant } from "@/lib/modelGroups";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  job: JobRecord | null;
  // Sibling checkpoints of one imported run; when given, the card shows a
  // checkpoint selector (in its parameter grid) that switches the evaluated job.
  group?: JobRecord[];
}

const EvaluationModal: React.FC<Props> = ({ open, onOpenChange, job, group }) => {
  const members = group && group.length > 0 ? group : null;
  const [activeId, setActiveId] = useState<string | null>(job?.id ?? null);
  useEffect(() => {
    setActiveId(job?.id ?? null);
  }, [job?.id]);
  const activeJob = members
    ? members.find((m) => m.id === activeId) ?? members[0]
    : job;
  const title = activeJob
    ? members
      ? modelBaseName(activeJob.name)
      : activeJob.name
    : "";
  const variants: ModelVariant[] | undefined = members
    ? members.map((m) => ({ step: modelStepOf(m.name), jobId: m.id }))
    : undefined;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[860px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            模型评测{title ? ` · ${title}` : ""}
          </DialogTitle>
        </DialogHeader>
        {activeJob ? (
          <div className="space-y-4">
            <EvaluationCard
              key={activeJob.id}
              job={activeJob}
              variants={variants}
              onSelectJob={setActiveId}
            />
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

export default EvaluationModal;
