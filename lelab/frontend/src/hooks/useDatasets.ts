import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { DatasetItem, listDatasets } from "@/lib/replayApi";

// Module-level cache keyed by API base so re-mounting a page (e.g. navigating
// back to Landing) renders the last-known datasets instantly instead of
// blocking on a fresh fetch. The fetch still runs and updates in the background.
const cache = new Map<string, DatasetItem[]>();

export const useDatasets = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [datasets, setDatasets] = useState<DatasetItem[]>(() => cache.get(baseUrl) ?? []);
  const [loading, setLoading] = useState(() => !cache.has(baseUrl));

  const refresh = useCallback(() => {
    // Only spin on a cold start; a warm revalidate updates silently.
    if (!cache.has(baseUrl)) setLoading(true);
    listDatasets(baseUrl, fetchWithHeaders)
      .then((d) => {
        cache.set(baseUrl, d);
        setDatasets(d);
      })
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { datasets, loading, refresh };
};
