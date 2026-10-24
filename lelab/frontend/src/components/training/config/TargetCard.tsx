import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps } from "../types";
import {
  RunnerFlavor,
  SSHConfig,
  SSHConfigInput,
  RemoteEnvInfo,
  InstallProgress,
  createSSHConfig,
  updateSSHConfig,
  deleteSSHConfig,
  testSSHConfig,
  checkSSHRemoteEnv,
  installLerobotRemote,
} from "@/lib/jobsApi";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Plus, Trash2, Wifi, Server, Download, CheckCircle, XCircle, Pencil, Check, AlertTriangle } from "lucide-react";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
  sshConfigs: SSHConfig[];
  sshConfigsLoading: boolean;
}

const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator ? f.accelerator : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

const emptySSHForm: SSHConfigInput = {
  name: "",
  host: "",
  port: 22,
  username: "",
  password: "",
  key_path: "",
  remote_python: "",
  remote_workdir: "",
  wandb_api_key: "",
  use_cn_mirror: true,
};

/** Parse `ssh [-p PORT] [USER@]HOST` into {host, port, username}. */
const parseSSHCommand = (cmd: string): { host: string; port: number; username: string } => {
  const trimmed = cmd.trim();
  if (!trimmed) return { host: "", port: 22, username: "" };
  // Match: ssh [-p PORT] [USER@]HOST
  const m = trimmed.match(/^ssh(?:\s+-p\s+(\d+))?\s+(?:([\w.-]+)@)?([\w.-]+)/i);
  if (!m) return { host: "", port: 22, username: "" };
  return {
    port: m[1] ? parseInt(m[1], 10) : 22,
    username: m[2] || "",
    host: m[3] || "",
  };
};

/** Generate a display name from SSH command: `root@106.75.44.90` or `106.75.44.90`. */
const sshCommandToName = (cmd: string): string => {
  const { host, username } = parseSSHCommand(cmd);
  if (!host) return "";
  return username ? `${username}@${host}` : host;
};

const STEP_LABELS = [
  "检测 GPU",
  "配置 pip",
  "安装 PyTorch",
  "安装 lerobot",
  "安装依赖",
  "验证",
];

const INSTALLABLE_POLICIES: { v: string; label: string }[] = [
  { v: "act", label: "ACT" },
  { v: "diffusion", label: "Diffusion" },
  { v: "pi0", label: "Pi0" },
  { v: "pi0_fast", label: "Pi0-Fast" },
  { v: "smolvla", label: "SmolVLA" },
  { v: "groot", label: "GR00T" },
  { v: "tdmpc", label: "TDMPC" },
  { v: "vqbet", label: "VQBet" },
  { v: "reward_classifier", label: "Reward" },
];

const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
  sshConfigs,
  sshConfigsLoading,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const target = config.target;
  const value =
    target.runner === "local"
      ? "local"
      : target.runner === "ssh"
        ? "ssh"
        : `hf:${target.flavor ?? ""}`;

  const [showSSHForm, setShowSSHForm] = useState(false);
  const [sshForm, setSSHForm] = useState<SSHConfigInput>(emptySSHForm);
  const [sshCommandText, setSSHCommandText] = useState("");
  const [sshCreating, setSSHCreating] = useState(false);
  const [editingSSHId, setEditingSSHId] = useState<string | null>(null);
  const [sshTesting, setSSHTesting] = useState<string | null>(null);
  const [envChecking, setEnvChecking] = useState(false);
  const [remoteEnv, setRemoteEnv] = useState<RemoteEnvInfo | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installProgress, setInstallProgress] = useState<InstallProgress | null>(null);
  const [installLogs, setInstallLogs] = useState<string[]>([]);
  const [useCnMirror, setUseCnMirror] = useState(false);
  const [installPolicies, setInstallPolicies] = useState<string[]>(
    INSTALLABLE_POLICIES.map((p) => p.v),
  );

  // Check for active SSH install task on mount (for reconnection)
  useEffect(() => {
    const checkActiveInstall = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/tasks/active`);
        if (!response.ok) return;
        const data = await response.json();
        const installTask = data.tasks?.find(
          (t: { task_type: string; status: string }) =>
            t.task_type === "ssh_install" && t.status === "running"
        );
        if (installTask) {
          setInstalling(true);
          setInstallProgress({
            step: installTask.metadata?.step || 1,
            total_steps: installTask.metadata?.total_steps || 6,
            label: installTask.label,
            status: "running",
            logs: [],
            message: installTask.message,
          });
        }
      } catch {
        // ignore
      }
    };
    checkActiveInstall();
  }, [baseUrl, fetchWithHeaders]);

  // Self-heal a frozen progress card: if the SSE stream dies (server restart,
  // network drop) the card would otherwise sit on a stale "running" step
  // forever. Reconcile against the server's task registry while we believe
  // an install is in flight. Query /tasks/all, not /tasks/active: a finished
  // install leaves the active list, and treating that as a dead stream
  // false-alarms every successful install.
  useEffect(() => {
    if (!installing) return;
    const id = setInterval(async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/tasks/all`);
        if (!response.ok) return;
        const data = await response.json();
        const task = (data.tasks ?? []).find(
          (t: { task_type: string; metadata?: Record<string, unknown> }) =>
            t.task_type === "ssh_install" &&
            String(t.metadata?.config_id ?? "") === String(target?.ssh_config_id ?? ""),
        );
        if (!task) {
          setInstalling(false);
          setInstallProgress(null);
          toast({
            title: "安装流中断",
            description:
              "与安装程序失去连接。请使用「检查环境」验证环境状态。",
            variant: "destructive",
          });
        } else if (task.status === "failed") {
          setInstalling(false);
        } else if (task.status === "done") {
          setInstalling(false);
          setInstallProgress(null);
          toast({
            title: "安装完成",
            description: "安装程序已完成。请使用「检查环境」确认环境状态。",
          });
        }
      } catch {
        // transient fetch error — keep waiting
      }
    }, 5000);
    return () => clearInterval(id);
  }, [installing, baseUrl, fetchWithHeaders, target?.ssh_config_id, toast]);

  const handleChange = (v: string) => {
    if (v === "local") {
      updateConfig("target", { runner: "local" });
    } else if (v === "ssh") {
      updateConfig("target", { runner: "ssh" });
    } else if (v.startsWith("hf:")) {
      const flavor = v.slice("hf:".length);
      updateConfig("target", { runner: "hf_cloud", flavor });
    }
  };

  const handleSSHConfigSelect = (configId: string) => {
    updateConfig("target", { runner: "ssh", ssh_config_id: configId });
  };

  const handleSaveSSH = async () => {
    if (!sshForm.name || !sshForm.host || !sshForm.username) {
      toast({
        title: "字段缺失",
        description: "请输入服务器名称和有效的 SSH 登录指令（需包含用户名和主机地址）",
        variant: "destructive",
      });
      return;
    }
    setSSHCreating(true);
    try {
      const input: SSHConfigInput = {
        ...sshForm,
        password: sshForm.password || undefined,
        key_path: sshForm.key_path || undefined,
        remote_workdir: sshForm.remote_workdir || undefined,
        wandb_api_key: sshForm.wandb_api_key || undefined,
        use_cn_mirror: true,
      };
      if (editingSSHId) {
        await updateSSHConfig(baseUrl, fetchWithHeaders, editingSSHId, input);
        toast({ title: "SSH 配置已更新", description: input.name });
      } else {
        const created = await createSSHConfig(baseUrl, fetchWithHeaders, input);
        toast({ title: "SSH 配置已保存", description: created.name });
        updateConfig("target", { runner: "ssh", ssh_config_id: created.id });
      }
      setSSHForm(emptySSHForm);
      setSSHCommandText("");
      setEditingSSHId(null);
      setShowSSHForm(false);
      window.location.reload();
    } catch (e) {
      toast({
        title: "保存失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setSSHCreating(false);
    }
  };

  const handleEditSSH = (cfg: SSHConfig) => {
    setEditingSSHId(cfg.id);
    setSSHForm({
      name: cfg.name,
      host: cfg.host,
      port: cfg.port,
      username: cfg.username,
      password: "",
      key_path: cfg.key_path ?? "",
      remote_python: cfg.remote_python,
      remote_workdir: cfg.remote_workdir ?? "",
      wandb_api_key: "",
      use_cn_mirror: cfg.use_cn_mirror ?? false,
    });
    setSSHCommandText(`ssh${cfg.port !== 22 ? ` -p ${cfg.port}` : ""} ${cfg.username ? `${cfg.username}@` : ""}${cfg.host}`);
    setShowSSHForm(true);
  };

  const handleCancelSSHForm = () => {
    setShowSSHForm(false);
    setEditingSSHId(null);
    setSSHForm(emptySSHForm);
    setSSHCommandText("");
  };

  const handleDeleteSSH = async (id: string) => {
    try {
      await deleteSSHConfig(baseUrl, fetchWithHeaders, id);
      toast({ title: "SSH 配置已删除" });
      if (target.ssh_config_id === id) {
        updateConfig("target", { runner: "ssh" });
      }
      window.location.reload();
    } catch (e) {
      toast({
        title: "删除失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleTestSSH = async (id: string) => {
    setSSHTesting(id);
    try {
      const result = await testSSHConfig(baseUrl, fetchWithHeaders, id);
      toast({
        title: result.success ? "连接成功" : "连接失败",
        description: result.message,
        variant: result.success ? "default" : "destructive",
      });
    } catch (e) {
      toast({
        title: "测试失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setSSHTesting(null);
    }
  };

  const handleCheckEnv = async (id: string) => {
    setEnvChecking(true);
    setRemoteEnv(null);
    try {
      const env = await checkSSHRemoteEnv(baseUrl, fetchWithHeaders, id);
      setRemoteEnv(env);
      if (!env.success) {
        toast({
          title: "环境检查失败",
          description: env.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "检查失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setEnvChecking(false);
    }
  };

  const handleInstallLerobot = async (id: string) => {
    setInstalling(true);
    setInstallProgress(null);
    setInstallLogs([]);
    try {
      const result = await installLerobotRemote(
        baseUrl,
        fetchWithHeaders,
        id,
        "",
        useCnMirror,
        (update) => {
          setInstallProgress(update);
          setInstallLogs((prev) => [...prev, ...update.logs.filter((l) => !prev.includes(l))]);
        },
        installPolicies,
      );
      toast({
        title: result.success ? "安装完成" : "安装失败",
        description: result.message,
        variant: result.success ? "default" : "destructive",
      });
      if (result.success) {
        await handleCheckEnv(id);
      }
    } catch (e) {
      toast({
        title: "安装失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setInstalling(false);
    }
  };

  const selectedSSH = sshConfigs.find((c) => c.id === target.ssh_config_id);

  useEffect(() => {
    if (selectedSSH) {
      setUseCnMirror(selectedSSH.use_cn_mirror ?? false);
    }
  }, [selectedSSH?.id]);

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="text-white">算力目标 Compute target</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <Label className="text-slate-300">训练运行在</Label>
          <Select value={value} onValueChange={handleChange}>
            <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
              <SelectValue placeholder={loading ? "加载中…" : "选择目标"} />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600 text-white">
              <SelectItem value="local">本地 — 本机（免费）</SelectItem>
              {flavors.map((f) => (
                <SelectItem
                  key={f.name}
                  value={`hf:${f.name}`}
                  disabled={!authenticated}
                >
                  {formatFlavorLine(f)}
                  {!authenticated && (
                    <span className="text-amber-300 ml-2 text-xs">
                      需登录 HF
                    </span>
                  )}
                </SelectItem>
              ))}
              <SelectItem value="ssh">
                <span className="flex items-center gap-1">
                  <Wifi className="w-3 h-3" /> SSH — 远程服务器
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-slate-500 mt-1">
            {target.runner === "ssh"
              ? "通过 SSH 在远程机器上训练；可远程检查/安装 lerobot。"
              : "所示费用为每小时运行价；训练完成后策略会上传到你的 HF 账号。"}
          </p>
        </div>

        {target.runner === "ssh" && (
          <div className="space-y-3 pt-2 border-t border-slate-700">
            <div className="flex items-center justify-between">
              <Label className="text-slate-300">SSH 连接</Label>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-slate-400 hover:text-white"
                onClick={() => showSSHForm ? handleCancelSSHForm() : (setEditingSSHId(null), setSSHForm(emptySSHForm), setSSHCommandText(""), setShowSSHForm(true))}
              >
                <Plus className="w-3 h-3 mr-1" />
                {showSSHForm ? "取消" : "新建连接"}
              </Button>
            </div>

            {showSSHForm && (
              <div className="space-y-2 bg-slate-900/50 rounded-lg p-3 border border-slate-700">
                <div>
                  <Label className="text-xs text-slate-400">服务器名称</Label>
                  <Input
                    value={sshForm.name}
                    onChange={(e) =>
                      setSSHForm({ ...sshForm, name: e.target.value })
                    }
                    placeholder=""
                    className="bg-slate-900 border-slate-600 text-white text-sm mt-0.5"
                  />
                </div>
                <div>
                  <Label className="text-xs text-slate-400">SSH 登录指令</Label>
                  <Input
                    value={sshCommandText}
                    onChange={(e) => {
                      const cmd = e.target.value;
                      setSSHCommandText(cmd);
                      const parsed = parseSSHCommand(cmd);
                      setSSHForm({
                        ...sshForm,
                        host: parsed.host || sshForm.host,
                        port: parsed.port || sshForm.port,
                        username: parsed.username || sshForm.username,
                      });
                    }}
                    placeholder="ssh -p 23 root@106.75.44.90"
                    className="bg-slate-900 border-slate-600 text-white text-sm mt-0.5 font-mono"
                  />
                </div>
                <div>
                  <Label className="text-xs text-slate-400">密码</Label>
                  <Input
                    type="password"
                    value={sshForm.password}
                    onChange={(e) =>
                      setSSHForm({ ...sshForm, password: e.target.value })
                    }
                    placeholder={editingSSHId ? "留空则保持当前密码" : ""}
                    className="bg-slate-900 border-slate-600 text-white text-sm mt-0.5"
                  />
                </div>
                <div>
                  <Label className="text-xs text-slate-400">wandb API key</Label>
                  <Input
                    type="password"
                    value={sshForm.wandb_api_key}
                    onChange={(e) =>
                      setSSHForm({ ...sshForm, wandb_api_key: e.target.value })
                    }
                    placeholder={editingSSHId ? "留空则保持当前 key" : ""}
                    className="bg-slate-900 border-slate-600 text-white text-sm mt-0.5"
                  />
                </div>
                <Button
                  onClick={handleSaveSSH}
                  disabled={sshCreating || !sshForm.host || !sshForm.password}
                  size="sm"
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
                >
                  {sshCreating ? (
                    <>
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" /> 保存中…
                    </>
                  ) : editingSSHId ? (
                    "更新连接"
                  ) : (
                    "保存连接"
                  )}
                </Button>
              </div>
            )}

            {sshConfigsLoading ? (
              <div className="text-xs text-slate-500 flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" /> 加载配置中…
              </div>
            ) : sshConfigs.length === 0 ? (
              <p className="text-xs text-slate-500">
                暂无保存的连接。点击"新建连接"添加一个。
              </p>
            ) : (
              <div className="space-y-1">
                <Select
                  value={target.ssh_config_id ?? ""}
                  onValueChange={handleSSHConfigSelect}
                >
                  <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg">
                    <SelectValue placeholder="选择连接" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600 text-white">
                    {sshConfigs.map((c) => (
                      <SelectItem key={c.id} value={c.id}>
                        <span className="flex items-center gap-2">
                          {c.name}
                          <span className="text-slate-500 text-xs">
                            {c.username}@{c.host}:{c.port}
                          </span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedSSH && (
                  <div className="space-y-2 pt-1">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs text-slate-400 hover:text-white"
                        onClick={() => handleEditSSH(selectedSSH)}
                      >
                        <Pencil className="w-3 h-3" />
                        <span className="ml-1">编辑</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs text-slate-400 hover:text-white"
                        onClick={() => handleTestSSH(selectedSSH.id)}
                        disabled={sshTesting === selectedSSH.id}
                      >
                        {sshTesting === selectedSSH.id ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Wifi className="w-3 h-3" />
                        )}
                        <span className="ml-1">测试</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs text-slate-400 hover:text-white"
                        onClick={() => handleCheckEnv(selectedSSH.id)}
                        disabled={envChecking}
                      >
                        {envChecking ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Server className="w-3 h-3" />
                        )}
                        <span className="ml-1">检查环境</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs text-red-400 hover:text-red-300"
                        onClick={() => handleDeleteSSH(selectedSSH.id)}
                      >
                        <Trash2 className="w-3 h-3" />
                        <span className="ml-1">删除</span>
                      </Button>
                    </div>

                    <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700 space-y-2">
                      <Label className="text-xs text-slate-400">
                        wandb API key（训练前自动登录）
                      </Label>
                      <div className="flex gap-2">
                        <Input
                          type="password"
                          value={sshForm.wandb_api_key || ""}
                          onChange={(e) =>
                            setSSHForm({ ...sshForm, wandb_api_key: e.target.value })
                          }
                          placeholder={selectedSSH.has_wandb_api_key ? "••••••••（已配置）" : "输入你的 wandb API key"}
                          className="bg-slate-900 border-slate-600 text-white text-sm flex-1"
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs border-green-600 text-green-400 hover:bg-green-600/20"
                          onClick={async () => {
                            if (!sshForm.wandb_api_key) return;
                            try {
                              await updateSSHConfig(baseUrl, fetchWithHeaders, selectedSSH.id, {
                                ...selectedSSH,
                                wandb_api_key: sshForm.wandb_api_key,
                              });
                              toast({
                                title: "已保存",
                                description: "wandb API key 已保存",
                              });
                              setSSHForm({ ...sshForm, wandb_api_key: "" });
                            } catch (e) {
                              toast({
                                title: "保存失败",
                                description: e instanceof Error ? e.message : String(e),
                                variant: "destructive",
                              });
                            }
                          }}
                        >
                          保存
                        </Button>
                      </div>
                      <p className="text-xs text-slate-500">
                        从{" "}
                        <a href="https://wandb.ai/authorize" target="_blank" rel="noreferrer" className="text-blue-400 hover:underline">
                          wandb.ai/authorize
                        </a>
                        {" "}获取你的 API key
                      </p>
                    </div>

                    {remoteEnv && remoteEnv.success && (
                      <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-slate-300">远程环境</span>
                          {remoteEnv.lerobot_installed ? (
                            <span className="flex items-center gap-1 text-xs text-green-400">
                              <CheckCircle className="w-3 h-3" /> lerobot {remoteEnv.lerobot_version}
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 text-xs text-amber-400">
                              <XCircle className="w-3 h-3" /> 未安装 lerobot
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                          <div className="text-slate-400">Python: <span className="text-slate-200">{remoteEnv.python_version || "unknown"}</span></div>
                          <div className="text-slate-400">CUDA: <span className={remoteEnv.cuda_available ? "text-green-400" : "text-slate-500"}>{remoteEnv.cuda_available ? remoteEnv.cuda_version || "yes" : "no"}</span></div>
                          {remoteEnv.gpu_name && <div className="text-slate-400 col-span-2">GPU: <span className="text-slate-200">{remoteEnv.gpu_name}</span></div>}
                          {remoteEnv.disk_free_gb != null && <div className="text-slate-400">Disk: <span className="text-slate-200">{remoteEnv.disk_free_gb} GB free</span></div>}
                          {remoteEnv.lerobot_installed && (
                            <div className="text-slate-400 col-span-2">
                              策略:{" "}
                              <span className={remoteEnv.policies_missing?.length ? "text-amber-400" : "text-green-400"}>
                                {(remoteEnv.policies_ready?.length ?? 0)}/9 可用
                              </span>
                              {remoteEnv.policies_missing?.length ? (
                                <span className="text-slate-500">，缺失: {remoteEnv.policies_missing.join(", ")}</span>
                              ) : null}
                            </div>
                          )}
                        </div>
                        {!!remoteEnv.issues?.length && (
                          <div className="space-y-1.5">
                            {remoteEnv.issues.map((iss) => (
                              <div
                                key={iss.code}
                                className={`rounded border px-2 py-1.5 text-[11px] leading-snug ${
                                  iss.severity === "blocking"
                                    ? "border-red-800/60 bg-red-950/30 text-red-200"
                                    : "border-amber-800/50 bg-amber-950/20 text-amber-200"
                                }`}
                              >
                                <div className="flex items-start gap-1.5">
                                  {iss.severity === "blocking" ? (
                                    <XCircle className="w-3 h-3 mt-0.5 shrink-0" />
                                  ) : (
                                    <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />
                                  )}
                                  <div>
                                    <div>{iss.message}</div>
                                    <div className="text-slate-400 mt-0.5">
                                      {iss.auto_fixable ? "可自动修复：" : "需手动处理："}
                                      {iss.fix}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                        <div className="space-y-2">
                          <div>
                            <div className="flex items-center justify-between">
                              <Label className="text-xs text-slate-400">
                                安装策略范围（只下载所选策略的依赖）
                                {!remoteEnv?.lerobot_installed && !installing && (
                                  <span className="text-slate-500"> — 点 Check env 查看已装状态</span>
                                )}
                              </Label>
                              <button
                                type="button"
                                className="text-xs text-blue-400 hover:text-blue-300"
                                onClick={() =>
                                  setInstallPolicies(
                                    installPolicies.length === INSTALLABLE_POLICIES.length
                                      ? []
                                      : INSTALLABLE_POLICIES.map((p) => p.v),
                                  )
                                }
                              >
                                {installPolicies.length === INSTALLABLE_POLICIES.length ? "清空" : "全选"}
                              </button>
                            </div>
                            <div className="grid grid-cols-3 gap-x-2 gap-y-1 mt-1">
                              {INSTALLABLE_POLICIES.map((p) => {
                                const installed = remoteEnv?.policies_ready?.includes(p.v);
                                const known =
                                  !!remoteEnv?.lerobot_installed &&
                                  (installed || remoteEnv?.policies_missing?.includes(p.v));
                                return (
                                  <label
                                    key={p.v}
                                    className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer"
                                  >
                                    <Checkbox
                                      checked={installPolicies.includes(p.v)}
                                      onCheckedChange={(c) =>
                                        setInstallPolicies((prev) =>
                                          c === true
                                            ? prev.includes(p.v) ? prev : [...prev, p.v]
                                            : prev.filter((x) => x !== p.v),
                                        )
                                      }
                                      className="border-slate-500 data-[state=checked]:bg-blue-600"
                                    />
                                    {p.label}
                                    {known && installed && (
                                      <span className="flex items-center gap-0.5 text-green-400">
                                        <CheckCircle className="w-3 h-3" />
                                        <span className="text-[10px]">已装</span>
                                      </span>
                                    )}
                                    {known && !installed && (
                                      <span className="text-[10px] text-amber-400">未装</span>
                                    )}
                                  </label>
                                );
                              })}
                            </div>
                            {installPolicies.length === 0 && (
                              <p className="text-[10px] text-amber-400 mt-1">
                                未勾选任何策略 — 将安装全部策略的 extra
                              </p>
                            )}
                          </div>
                          <div className="flex items-center gap-2 py-1">
                            <Checkbox
                              id="cn-mirror"
                              checked={useCnMirror}
                              onCheckedChange={(v) => setUseCnMirror(v === true)}
                              className="border-slate-500"
                            />
                            <Label htmlFor="cn-mirror" className="text-xs text-slate-400 cursor-pointer">
                              Use China mirrors (apt + pip: Aliyun, PyTorch: SJTU)
                            </Label>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full text-xs border-blue-600 text-blue-400 hover:bg-blue-600/20 hover:text-blue-300"
                            onClick={() => handleInstallLerobot(selectedSSH.id)}
                            disabled={installing}
                          >
                            {installing ? (
                              <>
                                <Loader2 className="w-3 h-3 mr-1 animate-spin" /> 安装中…
                              </>
                            ) : (
                              <>
                                <Download className="w-3 h-3 mr-1" />
                                {remoteEnv.lerobot_installed
                                  ? "重装 / 补装所选策略依赖"
                                  : "安装 lerobot"}
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    )}

                    {installing && installProgress && (
                      <div className="bg-slate-900/70 rounded-lg p-3 border border-slate-600 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-slate-300">
                            进度 {installProgress.step}/{installProgress.total_steps}
                          </span>
                          <span className={`text-xs ${
                            installProgress.status === "done" ? "text-green-400" :
                            installProgress.status === "failed" ? "text-red-400" :
                            "text-blue-400"
                          }`}>
                            {installProgress.status === "done" ? "完成" :
                             installProgress.status === "failed" ? "失败" :
                             installProgress.label}
                          </span>
                        </div>
                        {installProgress.message && (
                          <div className={`text-[10px] break-all ${
                            installProgress.status === "failed" ? "text-red-300" :
                            installProgress.status === "done" ? "text-green-300/80" :
                            "text-slate-400"
                          }`}>
                            {installProgress.message}
                          </div>
                        )}
                        {installProgress.hint && (
                          <div className="text-[10px] leading-snug text-amber-300/90 border-l-2 border-amber-700/60 pl-2">
                            {installProgress.hint}
                          </div>
                        )}
                        <div className="w-full bg-slate-700 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full transition-all duration-500 ${
                              installProgress.status === "failed" ? "bg-red-500" :
                              installProgress.status === "done" ? "bg-green-500" :
                              "bg-blue-500"
                            }`}
                            style={{ width: `${(installProgress.step / installProgress.total_steps) * 100}%` }}
                          />
                        </div>
                        <div className="grid grid-cols-6 gap-1">
                          {STEP_LABELS.map((label, i) => {
                            const n = i + 1;
                            const done = installProgress.step > n || (installProgress.step === n && installProgress.status === "done");
                            const active = installProgress.step === n && installProgress.status === "running";
                            const failed = installProgress.step === n && installProgress.status === "failed";
                            return (
                              <div key={label} className={`flex flex-col items-center gap-0.5 p-1 rounded text-center ${done ? "bg-green-900/30" : active ? "bg-blue-900/30" : failed ? "bg-red-900/30" : "bg-slate-800/50"}`}>
                                <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[8px] font-bold ${done ? "bg-green-600 text-white" : active ? "bg-blue-600 text-white" : failed ? "bg-red-600 text-white" : "bg-slate-600 text-slate-400"}`}>
                                  {done ? <Check className="w-2.5 h-2.5" /> : n}
                                </div>
                                <span className={`text-[8px] leading-tight ${done ? "text-green-400" : active ? "text-blue-400" : failed ? "text-red-400" : "text-slate-500"}`}>{label}</span>
                                {active && <Loader2 className="w-2.5 h-2.5 animate-spin text-blue-400" />}
                              </div>
                            );
                          })}
                        </div>
                        {installLogs.length > 0 && (
                          <div className="bg-black/40 rounded p-2 max-h-32 overflow-y-auto">
                            {installLogs.slice(-15).map((line, i) => (
                              <div key={i} className="text-[10px] text-slate-400 font-mono leading-relaxed truncate">{line}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TargetCard;
