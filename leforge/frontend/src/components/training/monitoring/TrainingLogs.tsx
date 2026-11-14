
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText } from 'lucide-react';
import { LogEntry } from '../types';

interface TrainingLogsProps {
  logs: LogEntry[];
  logContainerRef: React.RefObject<HTMLDivElement>;
}

const TrainingLogs: React.FC<TrainingLogsProps> = ({ logs, logContainerRef }) => {
  return (
    <Card className="bg-sky-50/70 border-sky-300 rounded-xl">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-slate-900">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-200">
            <FileText className="w-5 h-5 text-sky-700" />
          </div>
          Training Logs
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="bg-sky-100 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm border border-sky-300"
        >
          {logs.length === 0 ? (
            <div className="text-slate-600 py-8">
              No training logs yet. Start training to see output.
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className="text-slate-700 break-words whitespace-pre-wrap"
              >
                <span className="text-slate-600 mr-2 select-none">
                  {new Date(log.timestamp * 1000).toLocaleTimeString()}
                </span>
                {log.message}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TrainingLogs;
