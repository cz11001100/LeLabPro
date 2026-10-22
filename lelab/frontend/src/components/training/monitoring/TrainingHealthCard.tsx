import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, CheckCircle, TrendingDown, TrendingUp, Minus, Zap, Shield } from 'lucide-react';
import { useApi } from '@/contexts/ApiContext';

interface TrainingHealth {
  status: 'healthy' | 'warning' | 'critical';
  loss_trend: string;
  gradient_health: string;
  convergence_progress: number;
  overfitting_risk: 'low' | 'medium' | 'high';
  recommendations: string[];
  details: Record<string, any>;
}

interface TrainingHealthCardProps {
  jobId: string;
  refreshInterval?: number;
}

const TrainingHealthCard: React.FC<TrainingHealthCardProps> = ({
  jobId,
  refreshInterval = 10000,
}) => {
  const [health, setHealth] = useState<TrainingHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const { baseUrl, fetchWithHeaders } = useApi();

  useEffect(() => {
    let cancelled = false;

    const fetchHealth = async () => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/jobs/${jobId}/training-health`);
        if (!res.ok) throw new Error('Failed to fetch');
        const data = await res.json();
        if (!cancelled) setHealth(data);
      } catch {
        // Silently fail - health monitoring is optional
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, refreshInterval);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [baseUrl, fetchWithHeaders, jobId, refreshInterval]);

  if (loading || !health) {
    return (
      <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
        <CardContent className="p-6">
          <div className="text-slate-500 text-sm">Loading training health...</div>
        </CardContent>
      </Card>
    );
  }

  const statusConfig = {
    healthy: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-500/20', label: 'Healthy' },
    warning: { icon: AlertCircle, color: 'text-yellow-400', bg: 'bg-yellow-500/20', label: 'Warning' },
    critical: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500/20', label: 'Critical' },
  };

  const trendConfig = {
    decreasing: { icon: TrendingDown, color: 'text-green-400', label: 'Decreasing' },
    increasing: { icon: TrendingUp, color: 'text-red-400', label: 'Increasing' },
    plateau: { icon: Minus, color: 'text-yellow-400', label: 'Plateau' },
    oscillating: { icon: TrendingUp, color: 'text-orange-400', label: 'Oscillating' },
    insufficient_data: { icon: Minus, color: 'text-slate-400', label: 'Collecting data' },
  };

  const StatusIcon = statusConfig[health.status]?.icon || CheckCircle;
  const TrendIcon = trendConfig[health.loss_trend]?.icon || Minus;

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-white text-base">
          <div className="flex items-center gap-3">
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${statusConfig[health.status]?.bg} ${statusConfig[health.status]?.color}`}>
              <StatusIcon className="w-4 h-4" />
            </div>
            <span>Training Health</span>
          </div>
          <span className={`text-sm font-normal px-2 py-1 rounded ${statusConfig[health.status]?.bg} ${statusConfig[health.status]?.color}`}>
            {statusConfig[health.status]?.label}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="text-xs text-slate-400">Loss Trend</div>
            <div className="flex items-center gap-2">
              <TrendIcon className={`w-4 h-4 ${trendConfig[health.loss_trend]?.color}`} />
              <span className="text-sm text-white">{trendConfig[health.loss_trend]?.label}</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-xs text-slate-400">Gradient Health</div>
            <div className="flex items-center gap-2">
              <Zap className={`w-4 h-4 ${health.gradient_health === 'stable' ? 'text-green-400' : 'text-yellow-400'}`} />
              <span className="text-sm text-white capitalize">{health.gradient_health}</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-xs text-slate-400">Convergence</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all"
                  style={{ width: `${(health.convergence_progress * 100).toFixed(0)}%` }}
                />
              </div>
              <span className="text-sm text-white tabular-nums">{(health.convergence_progress * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-xs text-slate-400">Overfitting Risk</div>
            <div className="flex items-center gap-2">
              <Shield className={`w-4 h-4 ${
                health.overfitting_risk === 'low' ? 'text-green-400' :
                health.overfitting_risk === 'medium' ? 'text-yellow-400' : 'text-red-400'
              }`} />
              <span className="text-sm text-white capitalize">{health.overfitting_risk}</span>
            </div>
          </div>
        </div>

        {/* Details */}
        {health.details && (
          <div className="pt-3 border-t border-slate-700">
            <div className="grid grid-cols-2 gap-3 text-xs">
              {health.details.initial_loss != null && (
                <div>
                  <span className="text-slate-400">Initial Loss: </span>
                  <span className="text-white">{health.details.initial_loss.toFixed(4)}</span>
                </div>
              )}
              {health.details.current_loss != null && (
                <div>
                  <span className="text-slate-400">Current Loss: </span>
                  <span className="text-white">{health.details.current_loss.toFixed(4)}</span>
                </div>
              )}
              {health.details.min_loss != null && (
                <div>
                  <span className="text-slate-400">Min Loss: </span>
                  <span className="text-white">{health.details.min_loss.toFixed(4)}</span>
                </div>
              )}
              {health.details.avg_recent_grad != null && (
                <div>
                  <span className="text-slate-400">Avg Grad: </span>
                  <span className="text-white">{health.details.avg_recent_grad.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {health.recommendations && health.recommendations.length > 0 && (
          <div className="pt-3 border-t border-slate-700">
            <div className="text-xs text-slate-400 mb-2">Recommendations</div>
            <ul className="space-y-1">
              {health.recommendations.map((rec, idx) => (
                <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                  <span className="text-blue-400 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TrainingHealthCard;
