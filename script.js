// DOM Elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureElement = document.getElementById('gesture');
const wordElement = document.getElementById('word');
const sentenceElement = document.getElementById('sentence');

// State variables
let currentWord = '';
let currentSentence = '';
let lastGesture = '';
let lastGestureTime = 0;
let isPaused = false;
let isMuted = false;
const GESTURE_THRESHOLD = 1000;
const SPACE_GESTURE = 'open_hand';

// Models
let handposeModel = null;
let posenetModel = null;
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

// Speech synthesis setup
const synth = window.speechSynthesis;
let lastSpokenTime = 0;
const SPEECH_COOLDOWN = 2000;

// Load TensorFlow.js models
async function loadModels() {
  try {
    console.log('Loading TensorFlow.js models...');
    
    // Load handpose model
    handposeModel = await handpose.load();
    console.log('Handpose model loaded');
    
    // Load posenet model
    posenetModel = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 640, height: 480 },
      multiplier: 0.75
    });
    console.log('Posenet model loaded');
    
    return true;
  } catch (err) {
    console.error('Error loading TensorFlow models:', err);
    return false;
  }
}

// Camera setup
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Your browser does not support webcam access. Please try using Chrome or Firefox.");
    return false;
  }

  try {
    console.log('Requesting camera access...');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    console.log('Camera access granted');
    webcamElement.srcObject = stream;
    
    // Wait for the video to be ready
    await new Promise((resolve, reject) => {
      webcamElement.onloadedmetadata = () => {
        console.log('Video metadata loaded');
        resolve();
      };
      webcamElement.onerror = (error) => {
        console.error('Video element error:', error);
        reject(error);
      };
    });

    console.log('Camera setup complete');
    return true;
  } catch (err) {
    console.error("Error accessing webcam:", err);
    alert("Webcam access denied or not available. Please make sure your webcam is connected and you've granted permission.");
    return false;
  }
}

// Process hand tracking results
function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      // Draw hand landmarks and connections
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 2
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: '#FF0000',
        lineWidth: 1
      });
      
      // Detect gesture using both MediaPipe and TensorFlow
      if (!isPaused) {
        detectGestureWithBothModels(landmarks);
      }
    }
  }
  canvasCtx.restore();
}

// Detect gesture using both MediaPipe and TensorFlow
async function detectGestureWithBothModels(mediapipeLandmarks) {
  try {
    // Get MediaPipe gesture
    const mediapipeGesture = detectGestureMediaPipe(mediapipeLandmarks);
    
    // Get TensorFlow gesture
    const tfGesture = await detectGestureTensorFlow();
    
    // Combine results for better accuracy
    const finalGesture = combineGestureResults(mediapipeGesture, tfGesture);
    
    if (finalGesture) {
      updateGesture(finalGesture);
    }
  } catch (err) {
    console.error('Error in gesture detection:', err);
  }
}

// MediaPipe gesture detection
function detectGestureMediaPipe(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];
  
  const indexMCP = landmarks[5];
  const middleMCP = landmarks[9];
  const ringMCP = landmarks[13];
  const pinkyMCP = landmarks[17];
  
  const isThumbExtended = thumbTip.y < landmarks[3].y;
  const isIndexExtended = indexTip.y < indexMCP.y;
  const isMiddleExtended = middleTip.y < middleMCP.y;
  const isRingExtended = ringTip.y < ringMCP.y;
  const isPinkyExtended = pinkyTip.y < pinkyMCP.y;
  
  const indexAngle = calculateAngle(indexTip, indexMCP, middleMCP);
  const thumbAngle = calculateAngle(thumbTip, landmarks[3], indexMCP);
  
  // Basic gesture detection
  if (!isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended && 
      Math.abs(thumbAngle) < 30) {
    return 'A';
  }
  
  if (isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended && 
      Math.abs(indexAngle) > 45) {
    return 'B';
  }
  
  if (isIndexExtended && isMiddleExtended && !isRingExtended && !isPinkyExtended) {
    return 'C';
  }
  
  if (isThumbExtended && isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended) {
    return 'L';
  }
  
  if (!isIndexExtended && !isMiddleExtended && !isRingExtended && isPinkyExtended) {
    return 'I';
  }
  
  if (isThumbExtended && isPinkyExtended && !isIndexExtended && !isMiddleExtended && !isRingExtended) {
    return 'Y';
  }
  
  if (isIndexExtended && isMiddleExtended && isRingExtended && isPinkyExtended && !isThumbExtended) {
    return SPACE_GESTURE;
  }
  
  return null;
}

// TensorFlow gesture detection
async function detectGestureTensorFlow() {
  if (!handposeModel || !posenetModel) return null;
  
  try {
    const predictions = await handposeModel.estimateHands(webcamElement);
    const pose = await posenetModel.estimateSinglePose(webcamElement);
    
    if (predictions.length > 0) {
      const landmarks = predictions[0].landmarks;
      const fingerAngles = calculateFingerAnglesTensorFlow(landmarks);
      return classifyGestureTensorFlow(fingerAngles, pose);
    }
  } catch (err) {
    console.error('TensorFlow detection error:', err);
  }
  
  return null;
}

// Calculate finger angles using TensorFlow
function calculateFingerAnglesTensorFlow(landmarks) {
  const angles = {};
  
  const fingers = {
    thumb: [1, 2, 3, 4],
    index: [5, 6, 7, 8],
    middle: [9, 10, 11, 12],
    ring: [13, 14, 15, 16],
    pinky: [17, 18, 19, 20]
  };
  
  for (const [finger, joints] of Object.entries(fingers)) {
    angles[finger] = [];
    for (let i = 0; i < joints.length - 2; i++) {
      const angle = calculateAngle(
        landmarks[joints[i]],
        landmarks[joints[i + 1]],
        landmarks[joints[i + 2]]
      );
      angles[finger].push(angle);
    }
  }
  
  return angles;
}

// Classify gesture using TensorFlow
function classifyGestureTensorFlow(fingerAngles, pose) {
  const features = {
    thumbAngle: fingerAngles.thumb[0],
    indexAngle: fingerAngles.index[0],
    middleAngle: fingerAngles.middle[0],
    ringAngle: fingerAngles.ring[0],
    pinkyAngle: fingerAngles.pinky[0],
    handOrientation: calculateHandOrientation(pose)
  };
  
  if (features.thumbAngle > 45 && features.indexAngle < 30) {
    return 'A';
  } else if (features.indexAngle > 45 && features.middleAngle < 30) {
    return 'B';
  } else if (features.indexAngle > 45 && features.middleAngle > 45) {
    return 'C';
  } else if (features.thumbAngle < 30 && features.indexAngle > 45) {
    return 'L';
  } else if (features.pinkyAngle > 45 && features.thumbAngle < 30) {
    return 'I';
  } else if (features.thumbAngle > 45 && features.pinkyAngle > 45) {
    return 'Y';
  } else if (features.indexAngle > 45 && features.middleAngle > 45 && 
             features.ringAngle > 45 && features.pinkyAngle > 45) {
    return SPACE_GESTURE;
  }
  
  return null;
}

// Combine results from both models
function combineGestureResults(mediapipeGesture, tfGesture) {
  // If both models agree, use that gesture
  if (mediapipeGesture && tfGesture && mediapipeGesture === tfGesture) {
    return mediapipeGesture;
  }
  
  // If only one model detected a gesture, use it
  if (mediapipeGesture) return mediapipeGesture;
  if (tfGesture) return tfGesture;
  
  return null;
}

// Helper functions
function calculateAngle(A, B, C) {
  const BA = { x: A.x - B.x, y: A.y - B.y };
  const BC = { x: C.x - B.x, y: C.y - B.y };
  
  const dotProduct = BA.x * BC.x + BA.y * BC.y;
  const magnitudeBA = Math.sqrt(BA.x * BA.x + BA.y * BA.y);
  const magnitudeBC = Math.sqrt(BC.x * BC.x + BC.y * BC.y);
  
  const angle = Math.acos(dotProduct / (magnitudeBA * magnitudeBC));
  return angle * (180 / Math.PI);
}

function calculateHandOrientation(pose) {
  if (!pose || !pose.keypoints) return 0;
  
  const leftWrist = pose.keypoints.find(k => k.part === 'leftWrist');
  const rightWrist = pose.keypoints.find(k => k.part === 'rightWrist');
  
  if (leftWrist && rightWrist) {
    return Math.atan2(rightWrist.position.y - leftWrist.position.y, 
                     rightWrist.position.x - leftWrist.position.x);
  }
  
  return 0;
}

// Update gesture and word/sentence
function updateGesture(gesture) {
  const now = Date.now();
  
  if (now - lastGestureTime < GESTURE_THRESHOLD) {
    return;
  }
  
  gestureElement.textContent = gesture;
  
  if (gesture === SPACE_GESTURE) {
    if (currentWord.length > 0) {
      currentSentence += currentWord + ' ';
      sentenceElement.textContent = currentSentence;
      speakWord();
      currentWord = '';
      wordElement.textContent = '[empty]';
    }
    return;
  }
  
  if (gesture !== lastGesture && gesture !== 'unknown') {
    currentWord += gesture;
    wordElement.textContent = currentWord;
    lastGesture = gesture;
    lastGestureTime = now;
  }
}

// Speech functions
function speakWord() {
  if (isMuted || currentWord.length === 0) return;
  
  const now = Date.now();
  if (now - lastSpokenTime < SPEECH_COOLDOWN) return;
  
  const utterance = new SpeechSynthesisUtterance(currentWord);
  synth.speak(utterance);
  lastSpokenTime = now;
}

function speakSentence() {
  if (isMuted || currentSentence.length === 0) return;
  
  const utterance = new SpeechSynthesisUtterance(currentSentence);
  synth.speak(utterance);
}

// UI Control functions
function clearWord() {
  currentWord = '';
  wordElement.textContent = '[empty]';
  gestureElement.textContent = 'None';
}

function clearSentence() {
  currentSentence = '';
  sentenceElement.textContent = '[empty]';
}

function togglePause() {
  isPaused = !isPaused;
  const pauseBtn = document.getElementById('pause-btn');
  pauseBtn.textContent = isPaused ? '▶️ Resume Recognition' : '⏸️ Pause Recognition';
}

function toggleMute() {
  isMuted = !isMuted;
  const muteBtn = document.getElementById('mute-btn');
  muteBtn.textContent = isMuted ? '🔊 Unmute Speech' : '🔇 Mute Speech';
}

// Initialize the application
async function initialize() {
  try {
    console.log('Initializing application...');
    
    // Setup the camera
    const cameraReady = await setupCamera();
    if (!cameraReady) {
      throw new Error('Failed to setup camera');
    }
    
    // Load TensorFlow models
    const modelsReady = await loadModels();
    if (!modelsReady) {
      console.warn('TensorFlow models not loaded, using MediaPipe only');
    }
    
    // Setup MediaPipe
    hands.onResults(onResults);
    
    // Start hand tracking
    console.log('Starting hand tracking...');
    const camera = new Camera(webcamElement, {
      onFrame: async () => {
        if (!isPaused) {
          await hands.send({ image: webcamElement });
        }
      },
      width: 640,
      height: 480
    });
    camera.start();
    
    console.log('Application initialized successfully');
  } catch (err) {
    console.error('Initialization error:', err);
    alert('Failed to initialize application: ' + err.message);
  }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM loaded, starting initialization...');
  initialize();
});
