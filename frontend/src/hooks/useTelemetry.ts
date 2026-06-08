import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "../api";

export type TelemetryEvent = Record<string, unknown> & {
  event_type?: string;
  timestamp?: number;
};

export function useTelemetry(maxEvents = 2000) {
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const maxEventsRef = useRef(maxEvents);
  maxEventsRef.current = maxEvents;
  const reconnectAttemptRef = useRef(0);
  const prevLengthRef = useRef(0);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;
    // Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s
    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = setTimeout(connect, delay);
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    // Clean up old connection
    if (wsRef.current) {
      try {
        wsRef.current.onopen = null;
        wsRef.current.onclose = null;
        wsRef.current.onmessage = null;
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }

    const ws = new WebSocket(wsUrl("/ws/telemetry"));
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) {
        ws.close();
        return;
      }
      setConnected(true);
      reconnectAttemptRef.current = 0; // Reset backoff on successful connect
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      wsRef.current = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose will fire after onerror, triggering reconnect
    };

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as TelemetryEvent;
        // Batch events using length comparison to avoid closure stale references
        const maxLen = maxEventsRef.current;
        setEvents((prev) => {
          if (prev.length >= maxLen) {
            return [...prev.slice(-(maxLen - 1)), data];
          }
          return [...prev, data];
        });
        prevLengthRef.current += 1;
      } catch {
        /* ignore malformed */
      }
    };
  }, [scheduleReconnect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        try {
          wsRef.current.onopen = null;
          wsRef.current.onclose = null;
          wsRef.current.onmessage = null;
          wsRef.current.close();
        } catch {
          // ignore
        }
        wsRef.current = null;
      }
    };
  }, [connect]);

  const byType = useCallback(
    (type: string) => events.filter((e) => e.event_type === type),
    [events]
  );

  return { events, connected, byType };
}