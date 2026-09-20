import { useEffect, useRef, useState } from "react";
import {
  Cartesian3,
  Cartesian2,
  Color,
  createWorldTerrainAsync,
  Ion,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  Viewer,
} from "cesium";
import type { Entity, Viewer as ViewerType } from "cesium";

type GeoJsonFeature = {
  type: "Feature";
  id?: string;
  geometry: { type: string; coordinates: [number, number, number?] };
  properties: Record<string, unknown>;
};

type EventsResponse = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

const API_BASE_URL = import.meta.env.VITE_SKYNET_API_BASE_URL ?? "http://localhost:8080";
const ionToken = import.meta.env.VITE_CESIUM_ION_ACCESS_TOKEN?.trim();

function magnitudeRadius(magnitude: number): number {
  return Math.max(5, Math.min(22, 5 + Math.pow(2, magnitude)));
}

function formatTime(value: unknown): string {
  if (typeof value !== "string") return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function App() {
  const globeElement = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ViewerType | null>(null);
  const [eventCount, setEventCount] = useState(0);
  const [selected, setSelected] = useState<GeoJsonFeature | null>(null);
  const [status, setStatus] = useState("CONNECTING");
  const [terrainEnabled, setTerrainEnabled] = useState(false);

  useEffect(() => {
    if (!globeElement.current) return;

    let cancelled = false;
    const viewer = new Viewer(globeElement.current, {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      navigationHelpButton: false,
      sceneModePicker: false,
      selectionIndicator: false,
      timeline: false,
      shadows: false,
      shouldAnimate: false,
    });
    viewerRef.current = viewer;

    if (ionToken) {
      Ion.defaultAccessToken = ionToken;
      createWorldTerrainAsync()
        .then((terrain) => {
          if (!cancelled) {
            viewer.terrainProvider = terrain;
            setTerrainEnabled(true);
          }
        })
        .catch(() => setStatus("ONLINE / TERRAIN UNAVAILABLE"));
    }

    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((movement: { position: Cartesian2 }) => {
      const picked = viewer.scene.pick(movement.position);
      const entity = picked?.id as Entity | undefined;
      const feature = entity?.properties?.feature?.getValue?.();
      if (feature) setSelected(feature as GeoJsonFeature);
    }, ScreenSpaceEventType.LEFT_CLICK);

    async function loadEvents() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/events`);
        if (!response.ok) throw new Error(`Backend returned ${response.status}`);
        const data = (await response.json()) as EventsResponse;
        if (cancelled) return;

        viewer.entities.removeAll();
        for (const feature of data.features) {
          const [longitude, latitude, depth = 0] = feature.geometry.coordinates;
          const magnitude = Number(feature.properties.mag ?? 0);
          const entity = viewer.entities.add({
            id: feature.id,
            position: Cartesian3.fromDegrees(longitude, latitude, depth * 1000),
            point: {
              color: magnitude >= 5 ? Color.ORANGERED : Color.CYAN,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
              outlineColor: Color.WHITE.withAlpha(0.8),
              outlineWidth: 1,
              pixelSize: magnitudeRadius(magnitude),
            },
            properties: { feature },
          });
          entity.label = undefined;
        }
        setEventCount(data.features.length);
        setStatus("ONLINE");
      } catch (error) {
        console.error("Unable to load SkyNet events", error);
        setStatus("BACKEND UNAVAILABLE");
      }
    }

    void loadEvents();
    return () => {
      cancelled = true;
      handler.destroy();
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  return (
    <main className="app-shell">
      <div ref={globeElement} className="globe" />
      <header className="brand-panel">
        <div className="eyebrow">GLOBAL INTELLIGENCE PLATFORM</div>
        <h1>SKYNET</h1>
        <div className="status-line">
          <span className={`status-dot ${status.includes("ONLINE") ? "online" : ""}`} />
          <span>{status}</span>
          <span className="separator">•</span>
          <span>{eventCount} EARTHQUAKES</span>
          {terrainEnabled && <span className="terrain-badge">TERRAIN</span>}
        </div>
      </header>
      <div className="map-caption">INTERACTIVE 3D EARTH <span>•</span> LIVE USGS FEED</div>
      {selected && (
        <aside className="event-card">
          <button className="close-button" onClick={() => setSelected(null)} aria-label="Close">
            ×
          </button>
          <div className="card-kicker">USGS EARTHQUAKE</div>
          <div className="magnitude">{String(selected.properties.mag ?? "—")}</div>
          <div className="place">{String(selected.properties.place ?? "Unknown location")}</div>
          <dl>
            <div><dt>Depth</dt><dd>{String(selected.properties.depth ?? selected.geometry.coordinates[2] ?? "—")} km</dd></div>
            <div><dt>Time</dt><dd>{formatTime(selected.properties.time)}</dd></div>
            <div><dt>Event ID</dt><dd>{String(selected.properties.id ?? selected.id ?? "—")}</dd></div>
            <div><dt>Source</dt><dd>USGS</dd></div>
          </dl>
        </aside>
      )}
    </main>
  );
}

export default App;
