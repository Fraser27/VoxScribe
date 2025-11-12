import React, { createContext, useContext, useEffect, useState, useRef } from 'react';

const WebSocketContext = createContext(null);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [wsStt, setWsStt] = useState(null);
  const [wsTts, setWsTts] = useState(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const reconnectTimeoutRef = useRef(null);
  const heartbeatIntervalRef = useRef(null);

  const connectToService = (service) => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/${service}`;
      
      console.log(`Attempting WebSocket connection to ${service}:`, wsUrl);
      const websocket = new WebSocket(wsUrl);

      websocket.onopen = () => {
        console.log(`${service.toUpperCase()} WebSocket connected successfully`);
        if (service === 'stt') {
          setConnected(true);
        }
        startHeartbeat(websocket);
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages((prev) => [...prev, data]);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      websocket.onclose = (event) => {
        console.log(`${service.toUpperCase()} WebSocket disconnected:`, event.code, event.reason);
        if (service === 'stt') {
          setConnected(false);
        }
        stopHeartbeat();
        // Only reconnect if not a normal closure
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(() => connectToService(service), 5000);
        }
      };

      websocket.onerror = (error) => {
        console.warn(`${service.toUpperCase()} WebSocket error (this is normal if backend is not running):`, error);
      };

      if (service === 'stt') {
        setWsStt(websocket);
      } else if (service === 'tts') {
        setWsTts(websocket);
      }
    } catch (error) {
      console.error(`Failed to create ${service.toUpperCase()} WebSocket:`, error);
      // Retry connection after delay
      reconnectTimeoutRef.current = setTimeout(() => connectToService(service), 5000);
    }
  };

  const startHeartbeat = (websocket) => {
    heartbeatIntervalRef.current = setInterval(() => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  };

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  };

  useEffect(() => {
    // Connect to both STT and TTS WebSocket services
    connectToService('stt');
    connectToService('tts');

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopHeartbeat();
      if (wsStt) {
        wsStt.close();
      }
      if (wsTts) {
        wsTts.close();
      }
    };
  }, []);

  const clearMessages = () => setMessages([]);

  return (
    <WebSocketContext.Provider value={{ wsStt, wsTts, connected, messages, clearMessages }}>
      {children}
    </WebSocketContext.Provider>
  );
};
