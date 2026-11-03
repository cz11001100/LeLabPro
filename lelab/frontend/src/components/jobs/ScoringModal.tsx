import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import RolloutScoreCard from "@/components/jobs/RolloutScoreCard";

/**
 * Scoring from a dataset card. The full section also lives on the Dataset page;
 * this is the short path — jump straight to the dataset you are looking at
 * without leaving the job list.
 */
const ScoringModal: React.FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetRepoId?: string;
}> = ({ open, onOpenChange, datasetRepoId }) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[1100px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            数据集打分{datasetRepoId ? ` · ${datasetRepoId}` : ""}
          </DialogTitle>
        </DialogHeader>
        <RolloutScoreCard initialRepoId={datasetRepoId ?? ""} />
      </DialogContent>
    </Dialog>
  );
};

export default ScoringModal;
