import React from "react";
import HfAuthChip from "./HfAuthChip";

const LandingTopBar: React.FC = () => {
  return (
    <header className="sticky top-0 z-30 w-full border-b border-sky-200 bg-sky-100/95 backdrop-blur supports-[backdrop-filter]:bg-sky-100/70">
      <div className="mx-auto flex h-12 max-w-7xl items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <img
            src="/lovable-uploads/5e648747-34b7-4d8f-93fd-4dbd00aeeefc.png"
            alt="LeForge"
            className="h-7 w-7"
          />
          <span className="text-base font-semibold tracking-tight text-slate-900">
            LeForge
          </span>
        </div>
        <HfAuthChip />
      </div>
    </header>
  );
};

export default LandingTopBar;
