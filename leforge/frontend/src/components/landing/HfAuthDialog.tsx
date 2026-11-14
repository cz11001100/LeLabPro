import React, { useState } from "react";
import { Check, Copy, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface HfAuthDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const HfAuthDialog: React.FC<HfAuthDialogProps> = ({ open, onOpenChange }) => {
  const { auth, refetch } = useHfAuth();
  const [copied, setCopied] = useState(false);
  const [refetching, setRefetching] = useState(false);

  if (auth.status !== "unauthenticated") {
    return null;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(auth.loginCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.warn("Clipboard write failed:", err);
    }
  };

  const handleRefetch = async () => {
    setRefetching(true);
    try {
      await refetch();
    } finally {
      setRefetching(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-sky-100 border-sky-200 text-slate-900">
        <DialogHeader>
          <DialogTitle className="text-amber-700">
            Hugging Face CLI not configured
          </DialogTitle>
          <DialogDescription className="text-slate-600">
            Uploads, training, and replay-from-Hub require a logged-in HF CLI.
            Run this in a terminal:
          </DialogDescription>
        </DialogHeader>
        <pre className="bg-sky-50 p-3 rounded border border-sky-300 text-xs sm:text-sm overflow-x-auto flex items-center justify-between gap-2">
          <code className="text-green-700">{auth.loginCommand}</code>
          <button
            type="button"
            onClick={handleCopy}
            className="flex-shrink-0 text-slate-600 hover:text-slate-800 transition-colors"
            aria-label="Copy command"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-700" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </pre>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefetch}
          disabled={refetching}
          className="border-amber-300 bg-transparent text-amber-700 hover:bg-amber-100 hover:text-amber-50"
        >
          <RefreshCw
            className={`w-4 h-4 mr-2 ${refetching ? "animate-spin" : ""}`}
          />
          I've logged in — recheck
        </Button>
      </DialogContent>
    </Dialog>
  );
};

export default HfAuthDialog;
