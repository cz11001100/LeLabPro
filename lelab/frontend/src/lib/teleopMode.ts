/**
 * Global teleop input mode.
 *
 * Both teleoperation and dataset recording read this setting. The teleoperation
 * page is where the user switches it; the choice survives reloads and is sent
 * with every start request so the two flows stay in sync.
 */
export type TeleopMode = "leader" | "keyboard_ee";

const STORAGE_KEY = "lelab.teleopMode";

export const TELEOP_MODE_LABELS: Record<TeleopMode, string> = {
  leader: "Leader arm",
  keyboard_ee: "Keyboard (EE)",
};

export const getTeleopMode = (): TeleopMode => {
  try {
    return localStorage.getItem(STORAGE_KEY) === "keyboard_ee" ? "keyboard_ee" : "leader";
  } catch {
    return "leader";
  }
};

export const setTeleopMode = (mode: TeleopMode): void => {
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // Storage may be unavailable; the in-memory choice still applies to this page.
  }
};
