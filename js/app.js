/* ============================================
   EmotionAI — Main Application Logic
   Real-time LIVE webcam emotion detection
   ============================================ */

// Emotion configuration
const EMOTIONS = {
    happy:     { emoji: '😊', color: '#fbbf24', label: 'Happy' },
    sad:       { emoji: '😢', color: '#3b82f6', label: 'Sad' },
    angry:     { emoji: '😠', color: '#ef4444', label: 'Angry' },
    surprised: { emoji: '😲', color: '#f97316', label: 'Surprised' },
    fearful:   { emoji: '😨', color: '#8b5cf6', label: 'Fearful' },
    disgusted: { emoji: '🤢', color: '#22c55e', label: 'Disgusted' },
    neutral:   { emoji: '😐', color: '#94a3b8', label: 'Neutral' }
};

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights/';

// State
let video, canvas, ctx;
let isDetecting = false;
let showLandmarks = true;
let sessionStartTime = null;
let sessionTimerInterval = null;
let detectionInterval = null;
let emotionHistory = [];       // { timestamp, emotions, dominant }
let noFaceFrameCount = 0;

// Chart instances
let timelineChart = null;
let distributionChart = null;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    setupEventListeners();
});

function createParticles() {
    const container = document.getElementById('particles-container');
    const emojis = ['😊', '😢', '😠', '😲', '😨', '🤢', '😐', '🎭', '🧠', '⚡'];
    for (let i = 0; i < 25; i++) {
        const p = document.createElement('div');
        p.classList.add('particle');
        p.textContent = emojis[Math.floor(Math.random() * emojis.length)];
        p.style.left = Math.random() * 100 + '%';
        p.style.setProperty('--duration', (6 + Math.random() * 10) + 's');
        p.style.setProperty('--delay', (Math.random() * 8) + 's');
        p.style.fontSize = (1 + Math.random() * 1.5) + 'rem';
        container.appendChild(p);
    }
}

function setupEventListeners() {
    document.getElementById('start-btn').addEventListener('click', startApp);
    document.getElementById('stop-btn').addEventListener('click', stopApp);
    document.getElementById('screenshot-btn').addEventListener('click', takeScreenshot);
    document.getElementById('toggle-landmarks-btn').addEventListener('click', toggleLandmarks);
    document.getElementById('analytics-btn').addEventListener('click', () => showSection('analytics'));
    document.getElementById('back-btn').addEventListener('click', () => showSection('detection'));
}

// ============================================
// APP LIFECYCLE
// ============================================
async function startApp() {
    showLoadingOverlay();
    try {
        await loadModels();
        await startCamera();
        showSection('detection');
        hideLoadingOverlay();
        startDetectionLoop();
        startSessionTimer();
    } catch (err) {
        console.error('Error starting app:', err);
        alert('Error: ' + err.message + '\n\nMake sure you allow camera access and have a stable internet connection.');
        hideLoadingOverlay();
    }
}

function stopApp() {
    stopDetectionLoop();
    stopCamera();
    stopSessionTimer();
    showSection('landing');
}

// ============================================
// MODEL LOADING
// ============================================
async function loadModels() {
    const statusEl = document.getElementById('loading-status');
    const barEl = document.getElementById('loading-bar');

    statusEl.textContent = 'Loading Face Detection Model...';
    barEl.style.width = '10%';
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    document.getElementById('model-1').classList.add('loaded');
    barEl.style.width = '33%';

    statusEl.textContent = 'Loading Facial Landmark Model...';
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    document.getElementById('model-2').classList.add('loaded');
    barEl.style.width = '66%';

    statusEl.textContent = 'Loading Expression Recognition Model...';
    await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
    document.getElementById('model-3').classList.add('loaded');
    barEl.style.width = '100%';

    statusEl.textContent = 'All models loaded! Starting camera...';
}

// ============================================
// CAMERA
// ============================================
async function startCamera() {
    video = document.getElementById('webcam');
    canvas = document.getElementById('overlay');
    ctx = canvas.getContext('2d');

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });

    video.srcObject = stream;
    await new Promise(resolve => {
        video.onloadedmetadata = () => {
            video.play();
            resolve();
        };
    });

    // Match canvas to actual video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

function stopCamera() {
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

// ============================================
// REAL-TIME LIVE DETECTION LOOP
// Continuously analyses every frame from webcam
// ============================================
function startDetectionLoop() {
    isDetecting = true;
    noFaceFrameCount = 0;

    // Run detection every 100ms (~10 FPS) — LIVE, not photo-based
    detectionInterval = setInterval(async () => {
        if (!isDetecting || !video || video.paused || video.ended) return;
        await performDetection();
    }, 100);
}

function stopDetectionLoop() {
    isDetecting = false;
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

async function performDetection() {
    try {
        const detections = await faceapi
            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
            .withFaceLandmarks()
            .withFaceExpressions();

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (detections.length === 0) {
            noFaceFrameCount++;
            if (noFaceFrameCount > 10) {
                document.getElementById('no-face-msg').classList.remove('hidden');
            }
            updateFaceInfo(0, 0);
            return;
        }

        noFaceFrameCount = 0;
        document.getElementById('no-face-msg').classList.add('hidden');

        // Resize detections to canvas size
        const resized = faceapi.resizeResults(detections, {
            width: canvas.width,
            height: canvas.height
        });

        // Draw and process each face
        resized.forEach(detection => {
            drawFaceBox(detection);
            if (showLandmarks) drawLandmarks(detection);
            drawEmotionLabel(detection);
        });

        // Update UI with first face's expressions
        const expressions = resized[0].expressions;
        updateEmotionBars(expressions);
        updateDominantEmotion(expressions);
        trackHistory(expressions);
        updateFaceInfo(resized.length, resized[0].detection.score);

    } catch (err) {
        // Silently ignore detection errors during continuous loop
    }
}

// ============================================
// CANVAS DRAWING — Draws on live video feed
// ============================================
function drawFaceBox(detection) {
    const box = detection.detection.box;
    const score = detection.detection.score;
    const dominant = getDominant(detection.expressions);
    const color = EMOTIONS[dominant].color;

    // Neon glow box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.width, box.height, 8);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Corner accents
    const cornerLen = 15;
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + cornerLen);
    ctx.lineTo(box.x, box.y);
    ctx.lineTo(box.x + cornerLen, box.y);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - cornerLen, box.y);
    ctx.lineTo(box.x + box.width, box.y);
    ctx.lineTo(box.x + box.width, box.y + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + box.height - cornerLen);
    ctx.lineTo(box.x, box.y + box.height);
    ctx.lineTo(box.x + cornerLen, box.y + box.height);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - cornerLen, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height - cornerLen);
    ctx.stroke();
}

function drawLandmarks(detection) {
    const landmarks = detection.landmarks;
    const points = landmarks.positions;

    ctx.fillStyle = 'rgba(6, 182, 212, 0.6)';
    points.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw jawline, eyebrows, eyes, nose, mouth outlines
    const drawPath = (pts, close = false) => {
        ctx.strokeStyle = 'rgba(124, 58, 237, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);
        }
        if (close) ctx.closePath();
        ctx.stroke();
    };

    drawPath(landmarks.getJawOutline());
    drawPath(landmarks.getLeftEyeBrow());
    drawPath(landmarks.getRightEyeBrow());
    drawPath(landmarks.getLeftEye(), true);
    drawPath(landmarks.getRightEye(), true);
    drawPath(landmarks.getNose());
    drawPath(landmarks.getMouth(), true);
}

function drawEmotionLabel(detection) {
    const box = detection.detection.box;
    const dominant = getDominant(detection.expressions);
    const emotion = EMOTIONS[dominant];
    const confidence = Math.round(detection.expressions[dominant] * 100);
    const text = `${emotion.emoji} ${emotion.label} ${confidence}%`;

    const fontSize = 14;
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    const textWidth = ctx.measureText(text).width;
    const padding = 8;
    const bgHeight = fontSize + padding * 2;
    const bgWidth = textWidth + padding * 2;
    const bgX = box.x;
    const bgY = box.y - bgHeight - 6;

    // Background pill
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.beginPath();
    ctx.roundRect(bgX, bgY, bgWidth, bgHeight, 6);
    ctx.fill();

    // Border
    ctx.strokeStyle = emotion.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(bgX, bgY, bgWidth, bgHeight, 6);
    ctx.stroke();

    // Text
    ctx.fillStyle = '#fff';
    ctx.fillText(text, bgX + padding, bgY + padding + fontSize - 2);
}

// ============================================
// UI UPDATES
// ============================================
function updateEmotionBars(expressions) {
    Object.keys(EMOTIONS).forEach(emotion => {
        const value = Math.round((expressions[emotion] || 0) * 100);
        const bar = document.getElementById(`bar-${emotion}`);
        const val = document.getElementById(`val-${emotion}`);
        const item = bar.closest('.emotion-bar-item');

        bar.style.width = value + '%';
        val.textContent = value + '%';

        // Highlight active
        if (value > 50) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

function updateDominantEmotion(expressions) {
    const dominant = getDominant(expressions);
    const emotion = EMOTIONS[dominant];
    const confidence = Math.round(expressions[dominant] * 100);

    document.getElementById('dominant-emoji').textContent = emotion.emoji;
    document.getElementById('dominant-label').textContent = emotion.label;
    document.getElementById('dominant-confidence').textContent = confidence + '%';

    // Colorize display
    const display = document.getElementById('dominant-emotion-display');
    display.style.borderColor = emotion.color + '40';
    display.style.boxShadow = `0 0 15px ${emotion.color}20`;
}

function updateFaceInfo(count, score) {
    document.getElementById('face-count').textContent = count;
    document.getElementById('face-confidence').textContent = Math.round(score * 100) + '%';
    document.getElementById('landmark-status').textContent = count > 0 ? '68 pts' : '—';
}

function trackHistory(expressions) {
    const dominant = getDominant(expressions);
    const entry = {
        timestamp: Date.now(),
        emotions: { ...expressions },
        dominant: dominant
    };
    emotionHistory.push(entry);

    // Keep max 600 entries (~1 min at 10 FPS)
    if (emotionHistory.length > 600) {
        emotionHistory.shift();
    }
}

// ============================================
// SESSION TIMER
// ============================================
function startSessionTimer() {
    sessionStartTime = Date.now();
    sessionTimerInterval = setInterval(updateTimerDisplay, 1000);
}

function stopSessionTimer() {
    if (sessionTimerInterval) {
        clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    }
}

function updateTimerDisplay() {
    if (!sessionStartTime) return;
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    const timeStr = `${mins}:${secs}`;
    document.getElementById('session-timer').textContent = timeStr;
    const analyticsTimer = document.getElementById('analytics-timer');
    if (analyticsTimer) analyticsTimer.textContent = timeStr;
}

// ============================================
// NAVIGATION
// ============================================
function showSection(section) {
    const sections = ['landing', 'detection', 'analytics'];
    sections.forEach(s => {
        const el = document.getElementById(`${s}-section`);
        if (s === section) {
            el.classList.remove('hidden');
            el.classList.add('active');
        } else {
            el.classList.add('hidden');
            el.classList.remove('active');
        }
    });

    if (section === 'analytics') {
        updateAnalytics();
    }
}

function showLoadingOverlay() {
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoadingOverlay() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

// ============================================
// CONTROLS
// ============================================
function toggleLandmarks() {
    showLandmarks = !showLandmarks;
    const btn = document.getElementById('toggle-landmarks-btn');
    btn.classList.toggle('active', showLandmarks);
}

function takeScreenshot() {
    // Create a composite of video + overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw mirrored video
    tempCtx.save();
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(video, -tempCanvas.width, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.restore();

    // Draw overlay (already mirrored in CSS, but canvas data is not)
    tempCtx.save();
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(canvas, -tempCanvas.width, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.restore();

    // Download
    const link = document.createElement('a');
    link.download = `EmotionAI_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
    link.href = tempCanvas.toDataURL('image/png');
    link.click();

    // Show toast
    showToast();
}

function showToast() {
    const toast = document.getElementById('screenshot-toast');
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 2500);
}

// ============================================
// ANALYTICS
// ============================================
function updateAnalytics() {
    if (emotionHistory.length === 0) return;

    // Calculate stats
    const emotionCounts = {};
    const emotionTotals = {};
    Object.keys(EMOTIONS).forEach(e => { emotionCounts[e] = 0; emotionTotals[e] = 0; });

    emotionHistory.forEach(entry => {
        emotionCounts[entry.dominant]++;
        Object.keys(EMOTIONS).forEach(e => {
            emotionTotals[e] += (entry.emotions[e] || 0);
        });
    });

    // Dominant mood overall
    const dominantOverall = Object.keys(emotionCounts).reduce((a, b) =>
        emotionCounts[a] > emotionCounts[b] ? a : b
    );

    document.getElementById('stat-dominant').textContent =
        EMOTIONS[dominantOverall].emoji + ' ' + EMOTIONS[dominantOverall].label;
    document.getElementById('stat-samples').textContent = emotionHistory.length;

    // Average happiness
    const avgHappy = Math.round((emotionTotals.happy / emotionHistory.length) * 100);
    document.getElementById('stat-happiness').textContent = avgHappy + '%';

    // Mood stability (lower changes = more stable)
    let changes = 0;
    for (let i = 1; i < emotionHistory.length; i++) {
        if (emotionHistory[i].dominant !== emotionHistory[i - 1].dominant) changes++;
    }
    const stabilityPct = Math.round((1 - changes / Math.max(emotionHistory.length - 1, 1)) * 100);
    let stabilityLabel = 'Very Stable';
    if (stabilityPct < 40) stabilityLabel = 'Fluctuating';
    else if (stabilityPct < 60) stabilityLabel = 'Moderate';
    else if (stabilityPct < 80) stabilityLabel = 'Stable';
    document.getElementById('stat-stability').textContent = stabilityLabel;

    // Build charts
    buildTimelineChart();
    buildDistributionChart(emotionTotals);
}

function buildTimelineChart() {
    const chartCanvas = document.getElementById('timeline-chart');
    if (timelineChart) timelineChart.destroy();

    // Sample data points (max 50 for readability)
    const step = Math.max(1, Math.floor(emotionHistory.length / 50));
    const sampled = emotionHistory.filter((_, i) => i % step === 0);

    const labels = sampled.map((_, i) => {
        const elapsed = Math.floor((sampled[i].timestamp - sampled[0].timestamp) / 1000);
        return `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')}`;
    });

    const datasets = Object.keys(EMOTIONS).map(emotion => ({
        label: EMOTIONS[emotion].label,
        data: sampled.map(e => Math.round((e.emotions[emotion] || 0) * 100)),
        borderColor: EMOTIONS[emotion].color,
        backgroundColor: EMOTIONS[emotion].color + '20',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: false
    }));

    timelineChart = new Chart(chartCanvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 15 }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 10 },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    min: 0, max: 100,
                    ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });
}

function buildDistributionChart(totals) {
    const chartCanvas = document.getElementById('distribution-chart');
    if (distributionChart) distributionChart.destroy();

    const emotionKeys = Object.keys(EMOTIONS);
    const total = Object.values(totals).reduce((a, b) => a + b, 0) || 1;

    distributionChart = new Chart(chartCanvas, {
        type: 'doughnut',
        data: {
            labels: emotionKeys.map(e => EMOTIONS[e].label),
            datasets: [{
                data: emotionKeys.map(e => Math.round((totals[e] / total) * 100)),
                backgroundColor: emotionKeys.map(e => EMOTIONS[e].color),
                borderColor: '#0a0a1a',
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, padding: 12, boxWidth: 12 }
                }
            }
        }
    });
}

// ============================================
// UTILITIES
// ============================================
function getDominant(expressions) {
    return Object.keys(expressions).reduce((a, b) =>
        expressions[a] > expressions[b] ? a : b
    );
}
