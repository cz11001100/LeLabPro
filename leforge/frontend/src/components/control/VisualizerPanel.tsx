import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import UrdfViewer from "../UrdfViewer";
import Logo from "@/components/Logo";

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Optional content rendered as a column beside the 3D viewer (e.g. a camera panel). */
  rightSlot?: React.ReactNode;
  /** Optional controls rendered on the right side of the header row. */
  headerExtra?: React.ReactNode;
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  onGoBack,
  className,
  rightSlot,
  headerExtra,
}) => {
  return (
    <div
      className={cn(
        "w-full p-2 sm:p-4 space-y-4 lg:space-y-0 lg:space-x-4 flex flex-col lg:flex-row",
        className
      )}
    >
      <div className="bg-sky-100 rounded-lg p-4 flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onGoBack}
            className="text-slate-600 hover:text-slate-900 hover:bg-sky-200 flex-shrink-0"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <Logo iconOnly={true} />
          <div className="w-px h-6 bg-sky-200" />
          <h2 className="text-xl font-medium text-slate-800">Teleoperation</h2>
          {headerExtra && <div className="ml-auto flex items-center gap-2">{headerExtra}</div>}
        </div>
        <div className="flex-1 bg-sky-50 rounded border border-sky-200 min-h-[50vh] lg:min-h-0">
          <UrdfViewer />
        </div>
      </div>
      {rightSlot && (
        <div className="lg:w-96 flex flex-col">{rightSlot}</div>
      )}
    </div>
  );
};

export default VisualizerPanel;
