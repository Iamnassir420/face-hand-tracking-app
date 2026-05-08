import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const viewer = document.querySelector("#viewer");
const ctx = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const emptyState = document.querySelector("#emptyState");
const faceCount = document.querySelector("#faceCount");
const handCount = document.querySelector("#handCount");
const fps = document.querySelector("#fps");
const statusText = document.querySelector("#statusText");
const positionText = document.querySelector("#positionText");
const scaleText = document.querySelector("#scaleText");
const angleText = document.querySelector("#angleText");
const emotionText = document.querySelector("#emotionText");
const gestureText = document.querySelector("#gestureText");
const handSideText = document.querySelector("#handSideText");
const boxToggle = document.querySelector("#boxToggle");
const meshToggle = document.querySelector("#meshToggle");
const contourToggle = document.querySelector("#contourToggle");
const handToggle = document.querySelector("#handToggle");

let faceLandmarker;
let handLandmarker;
let stream;
let animationFrame = 0;
let lastVideoTime = -1;
let lastFrameAt = performance.now();
let smoothedFps = 0;

const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

const setStatus = (text) => {
  statusText.textContent = text;
};

const formatPercent = (value) => `${Math.round(value * 100)}%`;

const loadTrackers = async () => {
  if (faceLandmarker && handLandmarker) return;

  setStatus("Loading");
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  [faceLandmarker, handLandmarker] = await Promise.all([
    FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: FACE_MODEL_URL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 4,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    }),
    HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    })
  ]);

  setStatus("Ready");
};

const resizeCanvas = () => {
  const rect = viewer.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width: rect.width, height: rect.height };
};

const toScreenPoint = (landmark, width, height) => ({
  x: width - landmark.x * width,
  y: landmark.y * height
});

const getBounds = (landmarks, width, height) => {
  const points = landmarks.map((point) => toScreenPoint(point, width, height));
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);

  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
    centerX: left + (right - left) / 2,
    centerY: top + (bottom - top) / 2
  };
};

const getAngle = (matrixData) => {
  if (!matrixData?.length) return "-";
  const yaw = Math.atan2(matrixData[8], matrixData[10]) * (180 / Math.PI);
  const pitch = Math.atan2(-matrixData[9], Math.hypot(matrixData[8], matrixData[10])) * (180 / Math.PI);
  const yawLabel = yaw > 8 ? "right" : yaw < -8 ? "left" : "center";
  const pitchLabel = pitch > 8 ? "down" : pitch < -8 ? "up" : "level";
  return `${yawLabel}, ${pitchLabel}`;
};

const getBlendScore = (blendshapes, name) =>
  blendshapes?.categories?.find((category) => category.categoryName === name)?.score ?? 0;

const getEmotion = (blendshapes) => {
  if (!blendshapes) return "-";

  const smile = (getBlendScore(blendshapes, "mouthSmileLeft") + getBlendScore(blendshapes, "mouthSmileRight")) / 2;
  const frown = (getBlendScore(blendshapes, "mouthFrownLeft") + getBlendScore(blendshapes, "mouthFrownRight")) / 2;
  const jawOpen = getBlendScore(blendshapes, "jawOpen");
  const browUp = getBlendScore(blendshapes, "browInnerUp");
  const browDown =
    (getBlendScore(blendshapes, "browDownLeft") + getBlendScore(blendshapes, "browDownRight")) / 2;
  const eyeSquint =
    (getBlendScore(blendshapes, "eyeSquintLeft") + getBlendScore(blendshapes, "eyeSquintRight")) / 2;

  if (smile > 0.42 && jawOpen > 0.18) return `Laughing (${formatPercent(Math.min(1, smile + jawOpen / 2))})`;
  if (smile > 0.25) return `Happy (${formatPercent(smile)})`;
  if (jawOpen > 0.34 && browUp > 0.18) return `Surprised (${formatPercent((jawOpen + browUp) / 2)})`;
  if (frown > 0.16 || (browUp > 0.24 && smile < 0.12)) return `Sad (${formatPercent(Math.max(frown, browUp))})`;
  if (browDown > 0.22 && eyeSquint > 0.12) return `Focused (${formatPercent((browDown + eyeSquint) / 2)})`;
  return "Neutral";
};

const drawBoundingBox = (bounds) => {
  ctx.save();
  ctx.strokeStyle = "#30d6a1";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(48, 214, 161, 0.55)";
  ctx.shadowBlur = 14;
  ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
  ctx.restore();
};

const drawLandmarkDots = (landmarks, width, height) => {
  ctx.save();
  ctx.fillStyle = "rgba(255, 207, 90, 0.78)";
  for (let index = 0; index < landmarks.length; index += 5) {
    const point = toScreenPoint(landmarks[index], width, height);
    ctx.beginPath();
    ctx.arc(point.x, point.y, 1.45, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
};

const drawConnectionSet = (landmarks, connectors, width, height, color, lineWidth = 1) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();

  for (const connector of connectors) {
    const start = landmarks[connector.start ?? connector[0]];
    const end = landmarks[connector.end ?? connector[1]];
    if (!start || !end) continue;

    const a = toScreenPoint(start, width, height);
    const b = toScreenPoint(end, width, height);
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }

  ctx.stroke();
  ctx.restore();
};

const drawConnectors = (landmarks, width, height) => {
  drawConnectionSet(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    width,
    height,
    "rgba(102, 166, 255, 0.22)"
  );
  drawConnectionSet(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    width,
    height,
    "rgba(48, 214, 161, 0.95)",
    2
  );
  drawConnectionSet(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    width,
    height,
    "rgba(255, 255, 255, 0.78)"
  );
  drawConnectionSet(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    width,
    height,
    "rgba(255, 255, 255, 0.78)"
  );
  drawConnectionSet(
    landmarks,
    FaceLandmarker.FACE_LANDMARKS_LIPS,
    width,
    height,
    "rgba(255, 207, 90, 0.9)"
  );
};

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));

const getFingerState = (landmarks) => {
  const index = landmarks[8].y < landmarks[6].y;
  const middle = landmarks[12].y < landmarks[10].y;
  const ring = landmarks[16].y < landmarks[14].y;
  const pinky = landmarks[20].y < landmarks[18].y;
  const thumb =
    distance(landmarks[4], landmarks[0]) > distance(landmarks[3], landmarks[0]) &&
    Math.abs(landmarks[4].x - landmarks[5].x) > 0.035;

  return { thumb, index, middle, ring, pinky };
};

const getGesture = (landmarks) => {
  const fingers = getFingerState(landmarks);
  const raised = Object.values(fingers).filter(Boolean).length;
  const pinchDistance = distance(landmarks[4], landmarks[8]);

  if (pinchDistance < 0.045) return "Pinch";
  if (raised === 5) return "Open palm";
  if (raised === 0) return "Fist";
  if (fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) return "Peace";
  if (fingers.index && raised === 1) return "Pointing";
  if (fingers.thumb && raised === 1) return "Thumbs up";
  return "Hand visible";
};

const drawHand = (landmarks, width, height, label) => {
  drawConnectionSet(landmarks, HandLandmarker.HAND_CONNECTIONS, width, height, "rgba(255, 207, 90, 0.88)", 3);

  ctx.save();
  ctx.fillStyle = "#30d6a1";
  ctx.strokeStyle = "rgba(8, 10, 11, 0.74)";
  ctx.lineWidth = 3;

  for (const landmark of landmarks) {
    const point = toScreenPoint(landmark, width, height);
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
  }

  const wrist = toScreenPoint(landmarks[0], width, height);
  ctx.font = "700 14px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#eef3f4";
  ctx.strokeStyle = "rgba(8, 10, 11, 0.82)";
  ctx.lineWidth = 4;
  ctx.strokeText(label, wrist.x + 10, wrist.y - 12);
  ctx.fillText(label, wrist.x + 10, wrist.y - 12);
  ctx.restore();
};

const updateReadout = (results, width, height) => {
  const count = results.faceLandmarks.length;
  faceCount.textContent = String(count);

  if (!count) {
    positionText.textContent = "-";
    scaleText.textContent = "-";
    angleText.textContent = "-";
    emotionText.textContent = "-";
    return;
  }

  const bounds = getBounds(results.faceLandmarks[0], width, height);
  positionText.textContent = `${formatPercent(bounds.centerX / width)} x ${formatPercent(bounds.centerY / height)}`;
  scaleText.textContent = `${formatPercent(bounds.width / width)} wide`;
  angleText.textContent = getAngle(results.facialTransformationMatrixes?.[0]?.data);
  emotionText.textContent = getEmotion(results.faceBlendshapes?.[0]);
};

const updateHandReadout = (results) => {
  const hands = results.landmarks ?? [];
  handCount.textContent = String(hands.length);

  if (!hands.length) {
    gestureText.textContent = "-";
    handSideText.textContent = "-";
    return;
  }

  const gestures = hands.map((landmarks) => getGesture(landmarks));
  const sides = (results.handednesses ?? []).map((handedness) => handedness[0]?.displayName ?? "Hand");
  gestureText.textContent = gestures.join(", ");
  handSideText.textContent = sides.join(", ") || "-";
};

const renderFrame = () => {
  const { width, height } = resizeCanvas();
  ctx.clearRect(0, 0, width, height);

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const faceResults = faceLandmarker.detectForVideo(video, now);
    const handResults = handToggle.checked ? handLandmarker.detectForVideo(video, now) : { landmarks: [] };
    const instantFps = 1000 / Math.max(1, now - lastFrameAt);
    smoothedFps = smoothedFps ? smoothedFps * 0.85 + instantFps * 0.15 : instantFps;
    lastFrameAt = now;
    fps.textContent = String(Math.round(smoothedFps));

    if (contourToggle.checked) {
      for (const landmarks of faceResults.faceLandmarks) drawConnectors(landmarks, width, height);
    }

    for (const landmarks of faceResults.faceLandmarks) {
      const bounds = getBounds(landmarks, width, height);
      if (boxToggle.checked) drawBoundingBox(bounds);
      if (meshToggle.checked) drawLandmarkDots(landmarks, width, height);
    }

    if (handToggle.checked) {
      for (const [index, landmarks] of handResults.landmarks.entries()) {
        drawHand(landmarks, width, height, getGesture(landmarks));
      }
    }

    updateReadout(faceResults, width, height);
    updateHandReadout(handResults);
  }

  animationFrame = requestAnimationFrame(renderFrame);
};

const startCamera = async () => {
  try {
    startButton.disabled = true;
    setStatus("Starting");
    await loadTrackers();

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    emptyState.classList.add("hidden");
    stopButton.disabled = false;
    setStatus("Tracking");
    renderFrame();
  } catch (error) {
    console.error(error);
    setStatus("Blocked");
    startButton.disabled = false;
    stopButton.disabled = true;
    emptyState.classList.remove("hidden");
  }
};

const stopCamera = () => {
  cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  faceCount.textContent = "0";
  handCount.textContent = "0";
  fps.textContent = "0";
  positionText.textContent = "-";
  scaleText.textContent = "-";
  angleText.textContent = "-";
  emotionText.textContent = "-";
  gestureText.textContent = "-";
  handSideText.textContent = "-";
  emptyState.classList.remove("hidden");
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Ready");
};

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeCanvas);

lucide.createIcons();
resizeCanvas();
