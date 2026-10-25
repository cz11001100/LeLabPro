import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { JobRecord } from "@/lib/jobsApi";
import EvaluationCard from "@/components/jobs/EvaluationCard";
import RolloutScoreCard from "@/components/jobs/RolloutScoreCard";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  job: JobRecord | null;
}

const EvaluationModal: React.FC<Props> = ({ open, onOpenChange, job }) => (
  <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[860px] max-h-[90vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle className="text-white text-center text-2xl font-bold">
          模型评测{job ? ` · ${job.name}` : ""}
        </DialogTitle>
      </DialogHeader>
      {job ? (
        <div className="space-y-4">
          <EvaluationCard job={job} />
          <RolloutScoreCard />
        </div>
      ) : null}
    </DialogContent>
  </Dialog>
);

export default EvaluationModal;
