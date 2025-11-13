import { useState, useEffect } from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import SpaceBetween from '@cloudscape-design/components/space-between';
import ContentLayout from '@cloudscape-design/components/content-layout';
import Table from '@cloudscape-design/components/table';
import Button from '@cloudscape-design/components/button';
import Box from '@cloudscape-design/components/box';
import Badge from '@cloudscape-design/components/badge';
import Modal from '@cloudscape-design/components/modal';
import Alert from '@cloudscape-design/components/alert';
import Tabs from '@cloudscape-design/components/tabs';
import { useWebSocket } from '../contexts/WebSocketContext';

const ModelCatalogPage = () => {
  const [sttModels, setSttModels] = useState([]);
  const [ttsModels, setTtsModels] = useState([]);
  const [selectedItems, setSelectedItems] = useState([]);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [modelToDelete, setModelToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [downloading, setDownloading] = useState({});
  const { messages } = useWebSocket();

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    messages.forEach((msg) => {
      if (msg.type === 'download_progress') {
        const key = `${msg.engine}-${msg.model_id}`;
        setDownloading(prev => ({ ...prev, [key]: msg.progress }));
        
        if (msg.status === 'complete' || msg.status === 'error') {
          setTimeout(() => {
            setDownloading(prev => {
              const newState = { ...prev };
              delete newState[key];
              return newState;
            });
            loadModels();
          }, 2000);
        }
      } else if (msg.type === 'download_complete') {
        loadModels();
      }
    });
  }, [messages]);

  const loadModels = async () => {
    try {
      const [sttResponse, ttsResponse] = await Promise.all([
        fetch('/api/stt/models'),
        fetch('/api/tts/models')
      ]);

      if (sttResponse.ok) {
        const sttData = await sttResponse.json();
        setSttModels(sttData.models || []);
      }

      if (ttsResponse.ok) {
        const ttsData = await ttsResponse.json();
        setTtsModels(ttsData.models || []);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const handleDownload = async (engine, modelId, serviceType) => {
    const key = `${engine}-${modelId}`;
    setDownloading(prev => ({ ...prev, [key]: 0 }));

    try {
      const formData = new FormData();
      formData.append('engine', engine);
      formData.append('model_id', modelId);

      const endpoint = serviceType === 'stt' ? '/api/stt/download-model' : '/api/tts/download-model';
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Download failed');
      }
    } catch (error) {
      console.error('Download error:', error);
      alert(`Failed to start download: ${error.message}`);
      setDownloading(prev => {
        const newState = { ...prev };
        delete newState[key];
        return newState;
      });
    }
  };

  const handleDeleteClick = (model, serviceType) => {
    setModelToDelete({ ...model, serviceType });
    setDeleteModalVisible(true);
  };

  const handleDeleteConfirm = async () => {
    if (!modelToDelete) return;

    setDeleting(true);
    try {
      const endpoint = modelToDelete.serviceType === 'stt' 
        ? `/api/stt/models/${modelToDelete.engine}/${modelToDelete.model_id}`
        : `/api/tts/models/${modelToDelete.engine}/${modelToDelete.model_id}`;

      const response = await fetch(endpoint, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Delete failed');
      }

      await loadModels();
      setDeleteModalVisible(false);
      setModelToDelete(null);
    } catch (error) {
      console.error('Delete error:', error);
      alert(`Failed to delete model: ${error.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const getModelStatus = (model) => {
    const key = `${model.engine}-${model.model_id}`;
    if (downloading[key] !== undefined) {
      return <Badge color="blue">Downloading {downloading[key]}%</Badge>;
    }
    if (model.cached) {
      return <Badge color="green">Cached</Badge>;
    }
    return <Badge color="grey">Not Downloaded</Badge>;
  };

  const columnDefinitions = (serviceType) => [
    {
      id: 'engine',
      header: 'Engine',
      cell: item => item.engine,
      sortingField: 'engine'
    },
    {
      id: 'model_id',
      header: 'Model ID',
      cell: item => item.model_id,
      sortingField: 'model_id'
    },
    {
      id: 'display_name',
      header: 'Display Name',
      cell: item => item.display_name || item.model_id
    },
    {
      id: 'size',
      header: 'Size',
      cell: item => item.size
    },
    {
      id: 'status',
      header: 'Status',
      cell: item => getModelStatus(item)
    },
    {
      id: 'actions',
      header: 'Actions',
      cell: item => {
        const key = `${item.engine}-${item.model_id}`;
        const isDownloading = downloading[key] !== undefined;

        return (
          <SpaceBetween direction="horizontal" size="xs">
            {!item.cached && !isDownloading && (
              <Button
                variant="primary"
                iconName="download"
                onClick={() => handleDownload(item.engine, item.model_id, serviceType)}
              >
                Download
              </Button>
            )}
            {item.cached && (
              <Button
                variant="normal"
                iconName="remove"
                onClick={() => handleDeleteClick(item, serviceType)}
                disabled={isDownloading}
              >
                Delete
              </Button>
            )}
          </SpaceBetween>
        );
      }
    }
  ];

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Manage and download STT and TTS models"
        >
          Model Catalog
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Alert type="info">
          Download models to use them for transcription and synthesis. Cached models are stored locally and can be deleted to free up space.
        </Alert>

        <Tabs
          tabs={[
            {
              label: `STT Models (${sttModels.filter(m => m.cached).length}/${sttModels.length} cached)`,
              id: 'stt',
              content: (
                <Container>
                  <Table
                    columnDefinitions={columnDefinitions('stt')}
                    items={sttModels}
                    loadingText="Loading models"
                    sortingDisabled={false}
                    variant="embedded"
                    empty={
                      <Box textAlign="center" color="inherit">
                        <b>No STT models available</b>
                      </Box>
                    }
                  />
                </Container>
              )
            },
            {
              label: `TTS Models (${ttsModels.filter(m => m.cached).length}/${ttsModels.length} cached)`,
              id: 'tts',
              content: (
                <Container>
                  <Table
                    columnDefinitions={columnDefinitions('tts')}
                    items={ttsModels}
                    loadingText="Loading models"
                    sortingDisabled={false}
                    variant="embedded"
                    empty={
                      <Box textAlign="center" color="inherit">
                        <b>No TTS models available</b>
                      </Box>
                    }
                  />
                </Container>
              )
            }
          ]}
        />

        <Modal
          visible={deleteModalVisible}
          onDismiss={() => setDeleteModalVisible(false)}
          header="Delete Model"
          footer={
            <Box float="right">
              <SpaceBetween direction="horizontal" size="xs">
                <Button
                  variant="link"
                  onClick={() => setDeleteModalVisible(false)}
                  disabled={deleting}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={handleDeleteConfirm}
                  loading={deleting}
                >
                  Delete
                </Button>
              </SpaceBetween>
            </Box>
          }
        >
          {modelToDelete && (
            <SpaceBetween size="m">
              <Box>
                Are you sure you want to delete this model from cache?
              </Box>
              <Box variant="p">
                <strong>Engine:</strong> {modelToDelete.engine}
                <br />
                <strong>Model:</strong> {modelToDelete.model_id}
                <br />
                <strong>Size:</strong> {modelToDelete.size}
              </Box>
              <Alert type="warning">
                This will remove the model files from disk. You can re-download it later if needed.
              </Alert>
            </SpaceBetween>
          )}
        </Modal>
      </SpaceBetween>
    </ContentLayout>
  );
};

export default ModelCatalogPage;
