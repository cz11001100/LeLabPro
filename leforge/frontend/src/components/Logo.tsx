
import React from 'react';
import { cn } from '@/lib/utils';

interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({
  className,
  iconOnly = false
}) => {
  return <div className={cn("flex items-center gap-2", className)}>
      <img src="/lovable-uploads/5e648747-34b7-4d8f-93fd-4dbd00aeeefc.png" alt="LeForge Logo" className="h-8 w-8" />
      {!iconOnly && <span className="font-bold text-slate-900 text-2xl">LeForge</span>}
    </div>;
};

export default Logo;
