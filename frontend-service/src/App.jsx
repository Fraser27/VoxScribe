import React, { useState } from 'react';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import AppLayout from '@cloudscape-design/components/app-layout';
import TopNavigation from '@cloudscape-design/components/top-navigation';
import SideNavigation from '@cloudscape-design/components/side-navigation';
import STTPage from './pages/STTPage';
import TTSPage from './pages/TTSPage';
import ModelCatalogPage from './pages/ModelCatalogPage';
import { WebSocketProvider } from './contexts/WebSocketContext';

function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const [navigationOpen, setNavigationOpen] = useState(true);

  const navItems = [
    {
      type: 'section',
      text: 'Speech Services',
      items: [
        { type: 'link', text: 'Speech-to-Text', href: '/stt' },
        { type: 'link', text: 'Text-to-Speech', href: '/tts' }
      ]
    },
    {
      type: 'section',
      text: 'Management',
      items: [
        { type: 'link', text: 'Model Catalog', href: '/models' }
      ]
    }
  ];

  const activeHref = location.pathname;

  return (
    <WebSocketProvider>
      <TopNavigation
        identity={{
          href: '/',
          title: 'VoxScribe',
          logo: { src: '', alt: 'VoxScribe' }
        }}
        utilities={[
          {
            type: 'button',
            text: 'Documentation',
            href: '#',
            external: true
          }
        ]}
      />
      <AppLayout
        navigationOpen={navigationOpen}
        onNavigationChange={({ detail }) => setNavigationOpen(detail.open)}
        navigation={
          <SideNavigation
            activeHref={activeHref}
            header={{ text: 'Navigation', href: '/' }}
            items={navItems}
            onFollow={(event) => {
              if (!event.detail.external) {
                event.preventDefault();
                navigate(event.detail.href);
              }
            }}
          />
        }
        content={
          <Routes>
            <Route path="/" element={<Navigate to="/stt" replace />} />
            <Route path="/stt" element={<STTPage />} />
            <Route path="/tts" element={<TTSPage />} />
            <Route path="/models" element={<ModelCatalogPage />} />
          </Routes>
        }
        toolsHide
      />
    </WebSocketProvider>
  );
}

export default App;
