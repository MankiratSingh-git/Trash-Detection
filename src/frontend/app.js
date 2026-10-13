/**
 * Sortify ML Studio - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const modelSelect = document.getElementById('modelSelect');
  const modelMetaInfo = document.getElementById('modelMetaInfo');

  const confSlider = document.getElementById('confSlider');
  const confValue = document.getElementById('confValue');
  const iouSlider = document.getElementById('iouSlider');
  const iouValue = document.getElementById('iouValue');

  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const clearBtn = document.getElementById('clearBtn');
  const detectBtn = document.getElementById('detectBtn');
  const detectSpinner = document.getElementById('detectSpinner');
  const detectBtnText = document.getElementById('detectBtnText');

  const webcamBtn = document.getElementById('webcamBtn');
  const webcamContainer = document.getElementById('webcamContainer');
  const webcamVideo = document.getElementById('webcamVideo');
  const captureBtn = document.getElementById('captureBtn');
  const closeWebcamBtn = document.getElementById('closeWebcamBtn');

  const tabInteractive = document.getElementById('tabInteractive');
  const tabAnnotated = document.getElementById('tabAnnotated');
  const placeholderView = document.getElementById('placeholderView');
  const detectionCanvas = document.getElementById('detectionCanvas');
  const annotatedImgView = document.getElementById('annotatedImgView');

  const perfMetrics = document.getElementById('perfMetrics');
  const latencyBadge = document.getElementById('latencyBadge');
  const countBadge = document.getElementById('countBadge');

  const categorySummaryPills = document.getElementById('categorySummaryPills');
  const detectionsList = document.getElementById('detectionsList');
  const jsonOutput = document.getElementById('jsonOutput');
  const copyJsonBtn = document.getElementById('copyJsonBtn');

  // Application State
  const API_BASE = window.location.origin.startsWith('http')
    ? window.location.origin
    : 'http://localhost:8000';

  let currentFile = null;
  let currentImageBitmap = null;
  let currentResponse = null;
  let activeTab = 'interactive';
  let webcamStream = null;
  let highlightedIndex = -1;

  // 1. Initial Health & Model Discovery
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();
      
      statusPill.className = 'status-pill status-online';
      statusText.textContent = `Online • ${data.loaded_models.length ? data.loaded_models.join(', ') : 'Ready'}`;
      loadAvailableModels();
    } catch (err) {
      statusPill.className = 'status-pill status-offline';
      statusText.textContent = 'API Offline';
      console.warn('API is unreachable:', err);
    }
  }

  async function loadAvailableModels() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/models`);
      if (!res.ok) return;
      const models = await res.json();
      
      if (models && models.length > 0) {
        modelSelect.innerHTML = '';
        models.forEach((m) => {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.textContent = `${m.name.toUpperCase()} - ${m.description || m.task}`;
          modelSelect.appendChild(opt);
        });

        const active = models[0];
        modelMetaInfo.textContent = `${active.num_classes} Classes • ${active.device} • ${active.weights_path.split(/[\\/]/).pop()}`;
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  }

  // 2. Sliders
  confSlider.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  iouSlider.addEventListener('input', (e) => {
    iouValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // 3. File Input & Drag-and-Drop
  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, WebP).');
      return;
    }
    currentFile = file;
    detectBtn.disabled = false;
    highlightedIndex = -1;

    const img = new Image();
    img.onload = () => {
      currentImageBitmap = img;
      renderImagePreview(img);
    };
    img.src = URL.createObjectURL(file);
  }

  function renderImagePreview(img) {
    placeholderView.classList.add('hidden');
    annotatedImgView.classList.add('hidden');
    detectionCanvas.classList.remove('hidden');

    detectionCanvas.width = img.naturalWidth || img.width;
    detectionCanvas.height = img.naturalHeight || img.height;

    const ctx = detectionCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Reset results breakdown
    detectionsList.innerHTML = '<div class="empty-state">Ready to detect. Click "Detect & Classify Waste".</div>';
    categorySummaryPills.innerHTML = '';
    perfMetrics.classList.add('hidden');
  }

  clearBtn.addEventListener('click', () => {
    currentFile = null;
    currentImageBitmap = null;
    currentResponse = null;
    fileInput.value = '';
    detectBtn.disabled = true;

    detectionCanvas.classList.add('hidden');
    annotatedImgView.classList.add('hidden');
    placeholderView.classList.remove('hidden');

    detectionsList.innerHTML = '<div class="empty-state">Upload an image to start waste detection.</div>';
    categorySummaryPills.innerHTML = '';
    perfMetrics.classList.add('hidden');
    jsonOutput.querySelector('code').textContent = '{}';
  });

  // 4. Webcam Support
  webcamBtn.addEventListener('click', async () => {
    if (webcamStream) {
      stopWebcam();
      return;
    }
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      webcamVideo.srcObject = webcamStream;
      webcamContainer.classList.remove('hidden');
    } catch (err) {
      alert('Unable to access webcam: ' + err.message);
    }
  });

  captureBtn.addEventListener('click', () => {
    if (!webcamVideo.videoWidth) return;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = webcamVideo.videoWidth;
    offCanvas.height = webcamVideo.videoHeight;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(webcamVideo, 0, 0);

    offCanvas.toBlob((blob) => {
      const capturedFile = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
      handleSelectedFile(capturedFile);
      stopWebcam();
    }, 'image/jpeg', 0.95);
  });

  closeWebcamBtn.addEventListener('click', stopWebcam);

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }
    webcamContainer.classList.add('hidden');
  }

  // 5. Viewport Tabs
  tabInteractive.addEventListener('click', () => {
    activeTab = 'interactive';
    tabInteractive.classList.add('active');
    tabAnnotated.classList.remove('active');
    if (currentImageBitmap) {
      annotatedImgView.classList.add('hidden');
      detectionCanvas.classList.remove('hidden');
      drawCanvasWithDetections();
    }
  });

  tabAnnotated.addEventListener('click', () => {
    activeTab = 'annotated';
    tabAnnotated.classList.add('active');
    tabInteractive.classList.remove('active');
    if (currentResponse && currentResponse.annotated_image_base64) {
      detectionCanvas.classList.add('hidden');
      annotatedImgView.src = currentResponse.annotated_image_base64;
      annotatedImgView.classList.remove('hidden');
    } else if (currentImageBitmap) {
      alert('Run detection first to generate server annotations.');
    }
  });

  // 6. Run Inference
  detectBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    const selectedModel = modelSelect.value || 'yolov11';
    const conf = parseFloat(confSlider.value);
    const iou = parseFloat(iouSlider.value);

    // UI Loading state
    detectBtn.disabled = true;
    detectSpinner.classList.remove('hidden');
    detectBtnText.textContent = 'Detecting...';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('conf', conf.toString());
    formData.append('iou', iou.toString());
    formData.append('imgsz', '640');
    formData.append('return_image', 'true');

    try {
      const endpoint = `${API_BASE}/api/v1/${selectedModel}/predict`;
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      currentResponse = data;
      renderResults(data);
    } catch (err) {
      alert(`Detection failed: ${err.message}`);
      console.error(err);
    } finally {
      detectBtn.disabled = false;
      detectSpinner.classList.add('hidden');
      detectBtnText.textContent = 'Detect & Classify Waste';
    }
  });

  // 7. Render Detections and Canvas
  function renderResults(data) {
    // 1. Performance metrics
    perfMetrics.classList.remove('hidden');
    latencyBadge.textContent = `⚡ ${data.summary.inference_time_ms} ms`;
    countBadge.textContent = `🎯 ${data.summary.total_detections} items`;

    // 2. Category summary pills
    categorySummaryPills.innerHTML = '';
    const counts = data.summary.category_counts || {};
    Object.entries(counts).forEach(([cat, cnt]) => {
      const pill = document.createElement('span');
      pill.className = 'summary-pill';
      pill.textContent = `${cat}: ${cnt}`;
      categorySummaryPills.appendChild(pill);
    });

    // 3. Raw JSON Viewer
    jsonOutput.querySelector('code').textContent = JSON.stringify(data, null, 2);

    // 4. Draw Canvas Bounding Boxes
    drawCanvasWithDetections();

    // 5. Build Detections List Cards
    renderDetectionCards(data.detections);
  }

  function drawCanvasWithDetections() {
    if (!currentImageBitmap) return;

    detectionCanvas.width = currentImageBitmap.naturalWidth || currentImageBitmap.width;
    detectionCanvas.height = currentImageBitmap.naturalHeight || currentImageBitmap.height;
    const ctx = detectionCanvas.getContext('2d');

    // Draw base image
    ctx.drawImage(currentImageBitmap, 0, 0);

    if (!currentResponse || !currentResponse.detections) return;

    const detections = currentResponse.detections;
    const scale = Math.max(1, detectionCanvas.width / 1000);

    detections.forEach((det, idx) => {
      const isHighlighted = (idx === highlightedIndex);
      const box = det.box;
      const x1 = box.x1;
      const y1 = box.y1;
      const w = box.width;
      const h = box.height;
      const color = det.color || '#10B981';

      // Box border
      ctx.lineWidth = isHighlighted ? (4 * scale) : (2.5 * scale);
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, w, h);

      // Box semi-transparent fill on highlight
      if (isHighlighted) {
        ctx.fillStyle = `${color}33`;
        ctx.fillRect(x1, y1, w, h);
      }

      // Label background & text
      const label = `${det.class_name} • ${Math.round(det.confidence * 100)}%`;
      const fontSize = Math.max(12, Math.round(13 * scale));
      ctx.font = `600 ${fontSize}px 'Plus Jakarta Sans', sans-serif`;

      const textMetrics = ctx.measureText(label);
      const padX = 6 * scale;
      const padY = 4 * scale;
      const tagH = fontSize + padY * 2;
      const tagW = textMetrics.width + padX * 2;

      const tagY = Math.max(0, y1 - tagH);

      ctx.fillStyle = color;
      ctx.fillRect(x1, tagY, tagW, tagH);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x1 + padX, tagY + fontSize + (padY / 2));
    });
  }

  function renderDetectionCards(detections) {
    detectionsList.innerHTML = '';

    if (!detections || detections.length === 0) {
      detectionsList.innerHTML = '<div class="empty-state">No waste objects detected above the confidence threshold. Try lowering the threshold.</div>';
      return;
    }

    detections.forEach((det, idx) => {
      const card = document.createElement('div');
      card.className = 'detection-card';
      card.style.setProperty('--card-color', det.color || '#10B981');

      card.innerHTML = `
        <div class="card-top">
          <span class="item-name">${det.class_name.replace(/_/g, ' ')}</span>
          <span class="item-confidence">${Math.round(det.confidence * 100)}%</span>
        </div>
        <div>
          <span class="category-badge">${det.category || 'Waste'}</span>
        </div>
        <div class="guidance-box">
          <span class="guidance-icon">ℹ️</span>
          <span>${det.guidance || 'Dispose in appropriate container.'}</span>
        </div>
      `;

      // Highlight on card hover
      card.addEventListener('mouseenter', () => {
        highlightedIndex = idx;
        card.classList.add('active-highlight');
        drawCanvasWithDetections();
      });

      card.addEventListener('mouseleave', () => {
        highlightedIndex = -1;
        card.classList.remove('active-highlight');
        drawCanvasWithDetections();
      });

      detectionsList.appendChild(card);
    });
  }

  // 8. Copy JSON
  copyJsonBtn.addEventListener('click', () => {
    const text = jsonOutput.querySelector('code').textContent;
    navigator.clipboard.writeText(text).then(() => {
      copyJsonBtn.textContent = '✓ Copied';
      setTimeout(() => {
        copyJsonBtn.textContent = 'Copy JSON';
      }, 2000);
    });
  });

  // Run initial health check on load
  checkApiHealth();
  setInterval(checkApiHealth, 15000);
});
