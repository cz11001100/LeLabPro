export interface UrdfFileModel {
  path: string;
  blobUrl: string;
  name?: string;
}

export interface JointAnimationConfig {
  name: string;
  type: "sine" | "linear" | "constant";
  min: number;
  max: number;
  speed: number;
  offset: number;
  isDegrees?: boolean;
  customEasing?: (time: number) => number;
}

export interface RobotAnimationConfig {
  joints: JointAnimationConfig[];
  speedMultiplier?: number;
}
