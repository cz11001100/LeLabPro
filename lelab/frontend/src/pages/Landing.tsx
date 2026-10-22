import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { ChevronsUpDown, RotateCw, Rocket, Database, Brain, Loader2, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import LandingTopBar from "@/components/landing/LandingTopBar";
import Footer from "@/components/Footer";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import RecordingModal from "@/components/landing/RecordingModal";
import DatasetPicker from "@/components/landing/DatasetPicker";
import ResumeDatasetPicker from "@/components/landing/ResumeDatasetPicker";
import JobsSection from "@/components/jobs/JobsSection";
import InferenceModal from "@/components/landing/InferenceModal";
import EvaluationModal from "@/components/jobs/EvaluationModal";

import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { DatasetItem } from "@/lib/replayApi";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { isHostedSpace } from "@/lib/isHostedSpace";
import { useApi } from "@/contexts/ApiContext";
import { listJobs, JobRecord } from "@/lib/jobsApi";

const ON_SPACE = isHostedSpace();

const Landing = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const { auth } = useHfAuth();
  const location = useLocation();

  const {
    selectedName,
    selectedRecord,
    availableNames,
    isLoading: isLoadingRobots,
    selectRobot,
    createRobot,
    deleteRobot,
  } = useRobots();

  const { datasets, loading: datasetsLoading } = useDatasets();

  // Recording modal state
  const [showRecordingModal, setShowRecordingModal] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [singleTask, setSingleTask] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [episodeTimeS, setEpisodeTimeS] = useState(60);
  const [resetTimeS, setResetTimeS] = useState(15);
  const [streamingEncoding, setStreamingEncoding] = useState(true);
  const [resume, setResume] = useState(false);
  const [resumeRepoId, setResumeRepoId] = useState<string | null>(null); // Full repo_id for resume
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const [emaEnabled, setEmaEnabled] = useState(true);
  const [emaAlpha, setEmaAlpha] = useState(0.5);

  // Deploy modal state
  const { baseUrl, fetchWithHeaders } = useApi();
  const [importedModels, setImportedModels] = useState<JobRecord[]>([]);
  const [deployModalOpen, setDeployModalOpen] = useState(false);
  const [deployJob, setDeployJob] = useState<JobRecord | null>(null);
  const [deployStep, setDeployStep] = useState<number | null>(null);
  const [deployPopoverOpen, setDeployPopoverOpen] = useState(false);
  const [evalModalOpen, setEvalModalOpen] = useState(false);
  const [evalJob, setEvalJob] = useState<JobRecord | null>(null);
  const [evalPopoverOpen, setEvalPopoverOpen] = useState(false);

  useEffect(() => {
    listJobs(baseUrl, fetchWithHeaders, 100).then((jobs) => {
      setImportedModels(jobs.filter((j) => j.runner === "imported"));
    }).catch(() => {});
  }, [baseUrl, fetchWithHeaders]);

  const releaseStreamsRef = useRef<(() => void) | null>(null);

  const navigate = useNavigate();
  const { toast } = useToast();

  // Handle resume navigation from the Upload page: when arriving with
  // { resumeDataset: "user/name" }, pre-fill the modal for continuing that
  // dataset's recording session.
  const resumeHandledRef = useRef(false);
  useEffect(() => {
    const navState = location.state as { resumeDataset?: string; resumeTask?: string } | null;
    const resumeDatasetFromNav = navState?.resumeDataset;
    if (resumeDatasetFromNav && !resumeHandledRef.current && !datasetsLoading) {
      resumeHandledRef.current = true;
      setResumeRepoId(resumeDatasetFromNav);
      const namePart = resumeDatasetFromNav.includes("/")
        ? resumeDatasetFromNav.split("/")[1]
        : resumeDatasetFromNav;
      setDatasetName(namePart);
      if (navState?.resumeTask) {
        setSingleTask(navState.resumeTask);
      }
      setResume(true);
      setShowRecordingModal(true);
      window.history.replaceState({}, "");
    }
  }, [location.state, datasetsLoading]);

  // Clear camera state and release streams when returning to landing page
  useEffect(() => {
    if (cameras.length > 0) {
      console.log(
        "🧹 Landing page: Cleaning up camera state from previous session",
      );
      if (releaseStreamsRef.current) {
        releaseStreamsRef.current();
      }
      setCameras([]);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (releaseStreamsRef.current) {
        console.log("🧹 Landing page: Cleaning up camera streams on unmount");
        releaseStreamsRef.current();
      }
    };
  }, []);

  const openRecordingModal = () => {
    setCameras(selectedRecord ? [...(selectedRecord.cameras ?? [])] : []);
    setShowRecordingModal(true);
  };

  const handleRecordingModalClose = (open: boolean) => {
    setShowRecordingModal(open);
    if (!open && releaseStreamsRef.current) {
      console.log("🧹 Modal closed: Releasing camera streams");
      releaseStreamsRef.current();
    }
  };

  const handleTrainingClick = () => navigate("/training");

  const handleSSHTrainingClick = () =>
    navigate("/training", { state: { sshRunner: true } });

  const openHubViewer = (repoId: string, isPrivate: boolean) => {
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    const target = isPrivate
      ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
      : `https://huggingface.co${spacePath}`;
    window.open(target, "_blank", "noopener,noreferrer");
  };

  const handlePickExisting = (item: DatasetItem) => {
    if (item.source === "local" || item.source === "both") {
      // Navigate to upload page for existing local datasets
      navigate("/upload", {
        state: {
          datasetInfo: {
            dataset_repo_id: item.repo_id,
            source: item.source,
          },
        },
      });
      return;
    }
    openHubViewer(item.repo_id, item.private);
  };

  const handleResumeDataset = (item: DatasetItem) => {
    // Store the full repo_id for resume, and extract the name for display
    setResumeRepoId(item.repo_id);
    const namePart = item.repo_id.includes("/")
      ? item.repo_id.split("/")[1]
      : item.repo_id;
    setDatasetName(namePart);
    setResume(true);
    openRecordingModal();
  };

  const handleOpenCustom = (repoId: string) => {
    // Custom-typed repo IDs are always treated as Hub paths. We don't know
    // privacy, so route through the login redirect to be safe.
    openHubViewer(repoId, true);
  };

  const handleCreateDataset = (name: string) => {
    setDatasetName(name);
    setResume(false);
    setResumeRepoId(null); // Clear any previous resume repo_id
    openRecordingModal();
  };

  const handleStartRecording = async () => {
    if (!selectedRecord) {
      toast({
        title: "No robot selected",
        description: "Select or create a robot on the Landing page first.",
        variant: "destructive",
      });
      return;
    }
    const robot = selectedRecord;
    if (!robot.is_clean) {
      toast({
        title: "Robot not ready",
        description: `${robot.name} is missing a calibration. Configure it before recording.`,
        variant: "destructive",
      });
      return;
    }
    if (!datasetName || !singleTask) {
      toast({
        title: "Missing dataset details",
        description: "Please enter a dataset name and task description.",
        variant: "destructive",
      });
      return;
    }

    // When resuming, use the stored full repo_id to ensure we connect to the
    // exact same dataset (with timestamp suffix). Otherwise, construct from username.
    const datasetRepoId = resume && resumeRepoId
      ? resumeRepoId
      : (auth.status === "authenticated"
          ? `${auth.username}/${datasetName}`
          : datasetName);

    if (cameras.length > 0 && releaseStreamsRef.current) {
      console.log("🔓 Releasing camera streams before starting recording...");
      toast({
        title: "Preparing Camera Resources",
        description: `Releasing ${cameras.length} camera stream(s) for recording...`,
      });
      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.log("✅ Camera streams released, proceeding with recording...");
      toast({
        title: "Camera Resources Ready",
        description:
          "Camera streams released successfully. Starting recording...",
      });
    }

    const cameraDict = cameras.reduce(
      (acc, cam) => {
        acc[cam.name] = {
          type: cam.type,
          camera_index: cam.camera_index,
          width: cam.width,
          height: cam.height,
          fps: cam.fps,
          ...(cam.fourcc ? { fourcc: cam.fourcc } : {}),
          ...(cam.backend ? { backend: cam.backend } : {}),
        };
        return acc;
      },
      {} as Record<
        string,
        {
          type: string;
          camera_index?: number;
          width: number;
          height: number;
          fps?: number;
          fourcc?: string;
          backend?: string;
        }
      >,
    );

    const recordingConfig = {
      leader_port: robot.leader_port,
      follower_port: robot.follower_port,
      leader_config: robot.leader_config,
      follower_config: robot.follower_config,
      dataset_repo_id: datasetRepoId,
      single_task: singleTask,
      num_episodes: numEpisodes,
      episode_time_s: episodeTimeS,
      reset_time_s: resetTimeS,
      fps: 30,
      video: true,
      push_to_hub: false,
      resume: resume,
      streaming_encoding: streamingEncoding,
      cameras: cameraDict,
      ema_alpha: emaEnabled ? emaAlpha : null,
    };

    setShowRecordingModal(false);
    navigate("/recording", { state: { recordingConfig } });
  };

  return (
    <div
      className="min-h-screen bg-black text-white pb-16"
      style={{ ["--lelab-topbar-h" as string]: "48px" }}
    >
      <LandingTopBar />

      <div
        className="sticky z-20 bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/70 border-b border-gray-800"
        style={{ top: "var(--lelab-topbar-h)" }}
      >
        <div className="mx-auto max-w-7xl px-4 py-4 grid gap-3 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          <RobotConfigManager
            selectedName={selectedName}
            selectedRecord={selectedRecord}
            availableNames={availableNames}
            isLoading={isLoadingRobots}
            selectRobot={selectRobot}
            createRobot={createRobot}
            deleteRobot={deleteRobot}
            emaEnabled={emaEnabled}
            emaAlpha={emaAlpha}
          />
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-left h-10 flex items-center gap-2">
              <Database className="h-5 w-5 text-purple-400" />
              录制 Recording
            </h3>
            <DatasetPicker
              datasets={datasets}
              loading={datasetsLoading}
              onPickExisting={handlePickExisting}
              onOpenCustom={handleOpenCustom}
              onCreateNew={handleCreateDataset}
            >
              <Button
                variant="outline"
                role="combobox"
                className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
              >
                <span className="truncate text-gray-300 flex items-center gap-2">
                  {datasetsLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      加载数据集…
                    </>
                  ) : datasets.length === 0 ? (
                    "暂无数据集，点击创建…"
                  ) : (
                    "选择或创建数据集…"
                  )}
                </span>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </DatasetPicker>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-left h-10 flex items-center gap-2">
              <RotateCw className="h-5 w-5 text-blue-400" />
              续录 Resume Recording
            </h3>
            <ResumeDatasetPicker
              datasets={datasets}
              loading={datasetsLoading}
              onSelect={handleResumeDataset}
            >
              <Button
                variant="outline"
                className="w-full justify-between bg-gray-800 border-blue-600/50 text-white hover:bg-blue-900/20 hover:border-blue-500"
              >
                <span className="truncate text-blue-300 flex items-center gap-2">
                  {datasetsLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      加载数据集…
                    </>
                  ) : datasets.filter((d) => d.source === "local" || d.source === "both").length === 0 ? (
                    "暂无本地数据集，请先录制…"
                  ) : (
                    "选择数据集以继续…"
                  )}
                </span>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </ResumeDatasetPicker>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-left h-10 flex items-center gap-2">
              <Brain className="h-5 w-5 text-green-400" />
              模型 Model
            </h3>
            <Button
              onClick={handleTrainingClick}
              className="w-full bg-green-500 hover:bg-green-600 text-white"
            >
              训练 Training
            </Button>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-left h-10 flex items-center gap-2">
              <Rocket className="h-5 w-5 text-gray-400" />
              部署 Deploy
            </h3>
            <Popover open={deployPopoverOpen} onOpenChange={setDeployPopoverOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={deployPopoverOpen}
                  className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
                >
                  <span className="truncate text-gray-300 flex items-center gap-2">
                    {importedModels.length === 0 ? (
                      "暂无本地模型，请先导入…"
                    ) : (
                      "选择模型…"
                    )}
                  </span>
                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-gray-800 border-gray-600">
                <Command className="bg-gray-800 text-white">
                  <CommandList>
                    <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                      未找到模型。
                    </CommandEmpty>
                    <CommandGroup>
                      {importedModels.map((model) => (
                        <CommandItem
                          key={model.id}
                          value={model.name}
                          onSelect={() => {
                            setDeployJob(model);
                            setDeployStep(0);
                            setDeployModalOpen(true);
                            setDeployPopoverOpen(false);
                          }}
                          className="text-gray-300 hover:bg-gray-700 hover:text-white"
                        >
                          <span className="truncate">{model.name}</span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-left h-10 flex items-center gap-2">
              <FlaskConical className="h-5 w-5 text-purple-400" />
              评估 Evaluation
            </h3>
            <Popover open={evalPopoverOpen} onOpenChange={setEvalPopoverOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={evalPopoverOpen}
                  className="w-full justify-between bg-gray-800 border-purple-600/50 text-white hover:bg-purple-900/20 hover:border-purple-500"
                >
                  <span className="truncate text-purple-300 flex items-center gap-2">
                    {importedModels.length === 0 ? "暂无本地模型，请先导入…" : "选择模型以评测…"}
                  </span>
                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-gray-800 border-gray-600">
                <Command className="bg-gray-800 text-white">
                  <CommandList>
                    <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                      未找到模型。
                    </CommandEmpty>
                    <CommandGroup>
                      {importedModels.map((model) => (
                        <CommandItem
                          key={model.id}
                          value={model.name}
                          onSelect={() => {
                            setEvalJob(model);
                            setEvalModalOpen(true);
                            setEvalPopoverOpen(false);
                          }}
                          className="text-gray-300 hover:bg-gray-700 hover:text-white"
                        >
                          <span className="truncate">{model.name}</span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <JobsSection />
      </main>

      <Footer />

      <EvaluationModal
        open={evalModalOpen}
        onOpenChange={setEvalModalOpen}
        job={evalJob}
      />

      <UsageInstructionsModal
        open={showUsageModal}
        onOpenChange={setShowUsageModal}
        dismissible={!ON_SPACE}
      />

      <RecordingModal
        open={showRecordingModal}
        onOpenChange={handleRecordingModalClose}
        robot={selectedRecord}
        datasetName={datasetName}
        setDatasetName={setDatasetName}
        singleTask={singleTask}
        setSingleTask={setSingleTask}
        numEpisodes={numEpisodes}
        setNumEpisodes={setNumEpisodes}
        episodeTimeS={episodeTimeS}
        setEpisodeTimeS={setEpisodeTimeS}
        resetTimeS={resetTimeS}
        setResetTimeS={setResetTimeS}
        streamingEncoding={streamingEncoding}
        setStreamingEncoding={setStreamingEncoding}
        resume={resume}
        setResume={setResume}
        cameras={cameras}
        setCameras={setCameras}
        emaEnabled={emaEnabled}
        setEmaEnabled={setEmaEnabled}
        emaAlpha={emaAlpha}
        setEmaAlpha={setEmaAlpha}
        onStart={handleStartRecording}
        releaseStreamsRef={releaseStreamsRef}
      />

      {deployJob ? (
        <InferenceModal
          open={deployModalOpen}
          onOpenChange={setDeployModalOpen}
          robot={selectedRecord}
          jobId={deployJob.id}
          initialStep={deployStep}
        />
      ) : null}
    </div>
  );
};

export default Landing;
