// ============================================================
// APLICAȚIE DE ANALIZĂ FACIALĂ – script.js (v15, cu urechi)
// ============================================================
// - Bazat pe v14 (fără vârstă, fără OpenCV obligatoriu)
// - Adăugat detecție urechi folosind OpenCV.js (opțional) și
//   fallback geometric permanent.
// - Păstrează toate îmbunătățirile de culoare (white balance),
//   clasificatori calibrați.
// ============================================================

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_PATH =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// URL pentru OpenCV.js (opțional, doar pentru îmbunătățirea urechilor)
const OPENCV_URL = "https://docs.opencv.org/4.8.0/opencv.js";

let faceLandmarker = null;
let currentResults = null;
let objectUrls = [];
let frontalFile = null;
let profilFile = null;
let opencvReady = false;
let openCvLoadPromise = null;

// ============================================================
// INDICI LANDMARK-URI MEDIAPIPE FACE MESH (esențiali)
// ============================================================
const LM = {
    FACE_RIGHT_TEMPLE: 234,
    FACE_LEFT_TEMPLE: 454,
    RIGHT_JAW: 58,
    LEFT_JAW: 288,
    CHIN: 152,
    HAIRLINE_CENTER: 10,
    HAIRLINE_RIGHT: 67,
    HAIRLINE_LEFT: 297,
    FOREHEAD_CENTER: 151,
    RIGHT_CHEEKBONE: 50,
    LEFT_CHEEKBONE: 280,
    RIGHT_EYE_OUTER: 33,
    RIGHT_EYE_INNER: 133,
    LEFT_EYE_INNER: 362,
    LEFT_EYE_OUTER: 263,
    RIGHT_IRIS_CENTER: 468,
    LEFT_IRIS_CENTER: 473,
    RIGHT_BROW_OUTER: 70,
    RIGHT_BROW_INNER: 107,
    RIGHT_BROW_TOP: 65,
    LEFT_BROW_OUTER: 300,
    LEFT_BROW_INNER: 336,
    LEFT_BROW_TOP: 295,
    NOSE_TIP: 1,
    NOSE_BRIDGE_TOP: 6,
    NOSE_BRIDGE_MID: 168,
    NOSE_BRIDGE_BOTTOM: 2,
    RIGHT_NOSTRIL: 45,
    LEFT_NOSTRIL: 275,
    MOUTH_RIGHT: 61,
    MOUTH_LEFT: 291,
    MOUTH_TOP: 13,
    MOUTH_BOTTOM: 14,
    CHIN_RIGHT: 201,
    CHIN_LEFT: 200,
    CHIN_CREASE: 202,
    RIGHT_CHEEK_SKIN: 116,
    LEFT_CHEEK_SKIN: 345,
    FOREHEAD_SKIN: 8,
};

// ============================================================
// FUNCȚII UTILITARE
// ============================================================
function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function averageColor(colors) {
    if (!colors || colors.length === 0) return { r: 0, g: 0, b: 0 };
    const sum = colors.reduce((acc, c) => {
        acc.r += c.r;
        acc.g += c.g;
        acc.b += c.b;
        return acc;
    }, { r: 0, g: 0, b: 0 });
    return { r: sum.r / colors.length, g: sum.g / colors.length, b: sum.b / colors.length };
}

function medianColor(colors) {
    if (!colors || colors.length === 0) return { r: 0, g: 0, b: 0 };
    const rVals = colors.map(c => c.r).sort((a, b) => a - b);
    const gVals = colors.map(c => c.g).sort((a, b) => a - b);
    const bVals = colors.map(c => c.b).sort((a, b) => a - b);
    const mid = Math.floor(colors.length / 2);
    return { r: rVals[mid], g: gVals[mid], b: bVals[mid] };
}

function rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;
    if (max === min) { h = s = 0; } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return { h: h * 360, s: s * 100, l: l * 100 };
}

function classifyHairColor(avgDarkLum) {
    if (avgDarkLum < 40) return "Negru";
    if (avgDarkLum < 75) return "Șaten";
    if (avgDarkLum < 130) return "Blond";
    return "Cărunt";
}

function classifyEyeColor(r, g, b) {
    const { h, s, l } = rgbToHsl(r, g, b);
    if (l < 20 && s < 20) return "Negri";
    if (h >= 180 && h <= 260 && s > 10) return "Albaștri";
    if (h >= 60 && h < 160 && s > 10) return "Verzi";
    if ((h >= 10 && h < 50 && s > 15) || (l >= 20 && l < 60 && s > 20)) return "Căprui";
    if (l > 60) return "Albaștri";
    return "Căprui";
}

// ------------------------------------------------------------
// Normalizare white balance: calculează raportul canalelor față
// de culoarea pielii (landmark FOREHEAD_SKIN) și aplică corecția
// asupra unui pixel dat.
// ------------------------------------------------------------
function createColorNormalizer(canvas, ctx, landmarks) {
    const skinRef = landmarks[LM.FOREHEAD_SKIN];
    if (!skinRef) return (c) => c;
    const refSamples = samplePixelsAroundPoints(canvas, ctx, [skinRef], 1);
    if (refSamples.length === 0) return (c) => c;
    const refAvg = averageColor(refSamples);
    const avgLum = (refAvg.r + refAvg.g + refAvg.b) / 3;
    const scaleR = avgLum / (refAvg.r || 1);
    const scaleG = avgLum / (refAvg.g || 1);
    const scaleB = avgLum / (refAvg.b || 1);
    return (c) => ({
        r: clamp(Math.round(c.r * scaleR), 0, 255),
        g: clamp(Math.round(c.g * scaleG), 0, 255),
        b: clamp(Math.round(c.b * scaleB), 0, 255)
    });
}

function samplePixelsAroundPoints(canvas, ctx, points, radiusPx = 3) {
    if (!points || points.length === 0) return [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const pxPoints = points.map((p) => {
        const px = clamp(Math.round(p.x * canvas.width), 0, canvas.width - 1);
        const py = clamp(Math.round(p.y * canvas.height), 0, canvas.height - 1);
        minX = Math.min(minX, px - radiusPx);
        minY = Math.min(minY, py - radiusPx);
        maxX = Math.max(maxX, px + radiusPx);
        maxY = Math.max(maxY, py + radiusPx);
        return { px, py };
    });
    minX = clamp(minX, 0, canvas.width - 1);
    minY = clamp(minY, 0, canvas.height - 1);
    maxX = clamp(maxX, 0, canvas.width - 1);
    maxY = clamp(maxY, 0, canvas.height - 1);
    const boxW = Math.max(1, maxX - minX + 1);
    const boxH = Math.max(1, maxY - minY + 1);
    const imageData = ctx.getImageData(minX, minY, boxW, boxH);
    const data = imageData.data;
    const samples = [];
    for (const { px, py } of pxPoints) {
        for (let dx = -radiusPx; dx <= radiusPx; dx++) {
            for (let dy = -radiusPx; dy <= radiusPx; dy++) {
                const x = clamp(px + dx, minX, maxX) - minX;
                const y = clamp(py + dy, minY, maxY) - minY;
                const idx = (y * boxW + x) * 4;
                samples.push({ r: data[idx], g: data[idx + 1], b: data[idx + 2] });
            }
        }
    }
    return samples;
}

function luminance(c) { return (c.r + c.g + c.b) / 3; }

function localVariance(samples) {
    if (samples.length < 2) return 0;
    const lums = samples.map(luminance);
    const avg = lums.reduce((a, b) => a + b, 0) / lums.length;
    return lums.reduce((acc, v) => acc + (v - avg) ** 2, 0) / lums.length;
}

// ============================================================
// ORIENTARE POZĂ DE PROFIL (dreapta / stânga)
// ============================================================
function detectProfileSide(landmarks) {
    const rightZ = landmarks[LM.FACE_RIGHT_TEMPLE]?.z ?? 0;
    const leftZ = landmarks[LM.FACE_LEFT_TEMPLE]?.z ?? 0;
    return rightZ <= leftZ ? "right" : "left";
}

function getSideLandmarks(landmarks, side) {
    if (side === "left") {
        return {
            temple: landmarks[LM.FACE_LEFT_TEMPLE],
            jaw: landmarks[LM.LEFT_JAW],
            cheekbone: landmarks[LM.LEFT_CHEEKBONE],
        };
    }
    return {
        temple: landmarks[LM.FACE_RIGHT_TEMPLE],
        jaw: landmarks[LM.RIGHT_JAW],
        cheekbone: landmarks[LM.RIGHT_CHEEKBONE],
    };
}

// ============================================================
// ÎNCĂRCARE DINAMICĂ OPENCV.JS (opțional, cu cache)
// ============================================================
function loadOpenCV() {
    if (opencvReady && window.cv && window.cv.Mat) {
        return Promise.resolve();
    }
    if (openCvLoadPromise) {
        return openCvLoadPromise;
    }

    openCvLoadPromise = new Promise((resolve, reject) => {
        const waitForReady = () => {
            if (window.cv && window.cv.Mat) {
                opencvReady = true;
                resolve();
            } else {
                setTimeout(waitForReady, 100);
            }
        };

        const existing = document.querySelector(`script[src="${OPENCV_URL}"]`);
        if (existing) {
            waitForReady();
            return;
        }

        const script = document.createElement("script");
        script.src = OPENCV_URL;
        script.async = true;
        script.onload = waitForReady;
        script.onerror = () => {
            openCvLoadPromise = null;
            reject(new Error("Nu s-a putut încărca OpenCV.js."));
        };
        document.head.appendChild(script);
    });

    return openCvLoadPromise;
}

// ============================================================
// INIȚIALIZARE MEDIAPIPE FACE LANDMARKER
// ============================================================
async function initFaceLandmarker() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(WASM_PATH);
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
            runningMode: "IMAGE", numFaces: 1,
            outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false
        });
        console.log("FaceLandmarker inițializat cu succes (GPU)");
    } catch (gpuError) {
        console.warn("GPU delegate a eșuat, încerc CPU...", gpuError);
        try {
            const filesetResolver = await FilesetResolver.forVisionTasks(WASM_PATH);
            faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
                runningMode: "IMAGE", numFaces: 1,
                outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false
            });
            console.log("FaceLandmarker inițializat cu CPU");
        } catch (cpuError) {
            console.error("Eroare la inițializarea FaceLandmarker:", cpuError);
            throw cpuError;
        }
    }
}

// ============================================================
// GESTIONARE UPLOAD IMAGINI
// ============================================================
function replacePreviewUrl(preview, newUrl) {
    if (preview.src && preview.src.startsWith("blob:")) {
        const oldUrl = preview.src;
        objectUrls = objectUrls.filter((u) => u !== oldUrl);
        URL.revokeObjectURL(oldUrl);
    }
    objectUrls.push(newUrl);
    preview.src = newUrl;
}

function setupUploadZone(zoneId, fileInputId, previewId, removeBtnId, callback) {
    const zone = document.getElementById(zoneId);
    const fileInput = document.getElementById(fileInputId);
    const preview = document.getElementById(previewId);
    const removeBtn = document.getElementById(removeBtnId);

    zone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const url = URL.createObjectURL(file);
            replacePreviewUrl(preview, url);
            preview.classList.add("visible");
            removeBtn.classList.add("visible");
            callback(url, file);
        }
    });
    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        if (preview.src) {
            const url = preview.src;
            objectUrls = objectUrls.filter((u) => u !== url);
            URL.revokeObjectURL(url);
        }
        preview.src = "";
        preview.classList.remove("visible");
        removeBtn.classList.remove("visible");
        callback(null, null);
    });
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            fileInput.files = e.dataTransfer.files;
            const url = URL.createObjectURL(file);
            replacePreviewUrl(preview, url);
            preview.classList.add("visible");
            removeBtn.classList.add("visible");
            callback(url, file);
        }
    });
}

function setupUploads() {
    setupUploadZone("drop-frontal", "file-frontal", "preview-frontal", "remove-frontal", (url, file) => {
        frontalFile = file; checkAnalyzeButton();
    });
    setupUploadZone("drop-profil", "file-profil", "preview-profil", "remove-profil", (url, file) => {
        profilFile = file; checkAnalyzeButton();
    });
}

function checkAnalyzeButton() {
    const btn = document.getElementById("btn-analyze");
    btn.disabled = !frontalFile;
    const text = document.getElementById("analyze-text");
    text.textContent = frontalFile ? "Analizează fețele" : "Încarcă poza din față";
}

// ============================================================
// PROCESARE IMAGINE + EXTRAGERE LANDMARK-URI
// ============================================================
async function processImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        objectUrls.push(url);
        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            ctx.drawImage(img, 0, 0);
            resolve({ image: img, canvas, ctx, objectUrl: url });
        };
        img.onerror = () => reject(new Error("Nu s-a putut încărca imaginea."));
        img.src = url;
    });
}

async function extractLandmarks(imageData) {
    if (!faceLandmarker) await initFaceLandmarker();
    try {
        const result = await faceLandmarker.detect(imageData.image);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            const landmarks = result.faceLandmarks[0];
            if (landmarks.length >= 468) return landmarks;
            console.warn("Număr insuficient de landmark-uri:", landmarks.length);
        }
        return null;
    } catch (err) {
        console.error("Eroare la detectarea landmark-urilor:", err);
        return null;
    }
}

function purgeImageData(imageData) {
    if (!imageData) return;
    try {
        if (imageData.canvas) {
            const ctx = imageData.canvas.getContext("2d");
            if (ctx) ctx.clearRect(0, 0, imageData.canvas.width, imageData.canvas.height);
            imageData.canvas.width = 0; imageData.canvas.height = 0;
        }
        if (imageData.image) imageData.image.removeAttribute("src");
        if (imageData.objectUrl) {
            URL.revokeObjectURL(imageData.objectUrl);
            objectUrls = objectUrls.filter((u) => u !== imageData.objectUrl);
        }
    } catch (err) { console.warn("Curățare imagine: eroare minoră ignorată:", err); }
}

// ============================================================
// CLASIFICATORI PE CATEGORII
// ============================================================

// ---------- Frunte ----------
function classifyForehead(landmarks, faceWidth) {
    const hairline = landmarks[LM.HAIRLINE_CENTER];
    const browY = (landmarks[LM.RIGHT_BROW_TOP].y + landmarks[LM.LEFT_BROW_TOP].y) / 2;
    const faceHeight = distance(landmarks[LM.HAIRLINE_CENTER], landmarks[LM.CHIN]);
    const foreheadHeight = Math.max(0, hairline.y - browY);
    const heightRatio = foreheadHeight / faceHeight;
    const foreheadWidth = distance(landmarks[LM.HAIRLINE_RIGHT], landmarks[LM.HAIRLINE_LEFT]);
    const widthRatio = foreheadWidth / faceWidth;
    let result = [];
    if (widthRatio < 0.72) result.push("Îngustă");
    else if (widthRatio > 0.85) result.push("Lată");
    else result.push("Mijlocie");
    if (heightRatio > 0.32) result.push("Înaltă");
    else if (heightRatio < 0.22) result.push("Scundă");
    else if (!result.includes("Mijlocie")) result.push("Mijlocie");
    return { tip: result[0] || "Mijlocie", detalii: result.join(", "), raportLatime: widthRatio.toFixed(2), raportInaltime: heightRatio.toFixed(2) };
}

// ---------- Tipul feței ----------
function classifyFaceType(landmarks) {
    const foreheadW = distance(landmarks[LM.HAIRLINE_RIGHT], landmarks[LM.HAIRLINE_LEFT]);
    const cheekboneW = distance(landmarks[LM.FACE_RIGHT_TEMPLE], landmarks[LM.FACE_LEFT_TEMPLE]);
    const jawW = distance(landmarks[LM.RIGHT_JAW], landmarks[LM.LEFT_JAW]);
    const faceH = distance(landmarks[LM.HAIRLINE_CENTER], landmarks[LM.CHIN]);
    const ratio = faceH / cheekboneW, jawRatio = jawW / cheekboneW, foreheadRatio = foreheadW / cheekboneW;
    let tip;
    if (ratio > 1.45 && jawRatio < 0.75) tip = foreheadRatio > 0.82 ? "Triunghiulară" : "Ascuțită";
    else if (ratio > 1.35 && foreheadRatio > 0.88 && jawRatio > 0.82) tip = "Dreptunghiulară";
    else if (ratio > 1.25 && foreheadRatio < 0.85 && jawRatio < 0.80 && cheekboneW > foreheadW && cheekboneW > jawW) tip = "Romboidă";
    else if (ratio > 1.25) tip = "Ovală";
    else if (ratio < 1.15 && foreheadRatio > 0.88 && jawRatio > 0.85) tip = "Pătrată";
    else if (ratio < 1.25) tip = "Rotundă";
    else tip = "Ovală";
    return { tip, raport: ratio.toFixed(2), latimePometi: cheekboneW.toFixed(2), latimeMaxilar: jawW.toFixed(2) };
}

// ---------- Ochi ----------
function classifyEyes(landmarks, canvas, ctx, faceWidth, normalizer) {
    const rEyeW = distance(landmarks[LM.RIGHT_EYE_OUTER], landmarks[LM.RIGHT_EYE_INNER]);
    const lEyeW = distance(landmarks[LM.LEFT_EYE_OUTER], landmarks[LM.LEFT_EYE_INNER]);
    const avgEyeW = (rEyeW + lEyeW) / 2, eyeRatio = avgEyeW / faceWidth;
    let marime;
    if (eyeRatio < 0.15) marime = "Mici";
    else if (eyeRatio > 0.21) marime = "Mari";
    else marime = "Mijlocii";

    let culoare = "Nedeterminată";
    const irisPoints = [LM.RIGHT_IRIS_CENTER, LM.LEFT_IRIS_CENTER]
        .filter(idx => idx < landmarks.length)
        .map(idx => landmarks[idx]);
    if (irisPoints.length > 0) {
        const samples = samplePixelsAroundPoints(canvas, ctx, irisPoints, 2);
        if (samples.length > 0) {
            const normalized = samples.map(normalizer);
            const avg = averageColor(normalized);
            culoare = classifyEyeColor(avg.r, avg.g, avg.b);
        }
    }
    return { culoare, marime, raportOchi: eyeRatio.toFixed(2) };
}

// ---------- Gură ----------
function classifyMouth(landmarks, faceWidth) {
    const mouthW = distance(landmarks[LM.MOUTH_RIGHT], landmarks[LM.MOUTH_LEFT]);
    const mouthRatio = mouthW / faceWidth;
    let marime;
    if (mouthRatio < 0.30) marime = "Mică";
    else if (mouthRatio > 0.42) marime = "Mare";
    else marime = "Mijlocie";
    const cornerR = landmarks[LM.MOUTH_RIGHT], cornerL = landmarks[LM.MOUTH_LEFT];
    const cornerAvgY = (cornerR.y + cornerL.y) / 2, topY = landmarks[LM.MOUTH_TOP].y;
    const diff = topY - cornerAvgY;
    let colturi;
    if (diff > 0.015) colturi = "Colțuri ridicate";
    else if (diff < -0.015) colturi = "Colțuri coborâte";
    else colturi = "Liniară";
    return { colturi, marime, raportGura: mouthRatio.toFixed(2) };
}

// ---------- Bărbie ----------
function classifyChin(landmarks, faceWidth, canvas, ctx) {
    const chinW = distance(landmarks[LM.CHIN_RIGHT], landmarks[LM.CHIN_LEFT]);
    const chinRatio = chinW / faceWidth;
    let tip;
    if (chinRatio < 0.18) tip = "Ascuțită";
    else if (chinRatio > 0.30) tip = "Plată";
    else tip = "Normală";
    if (canvas && ctx && LM.CHIN_CREASE < landmarks.length) {
        const samples = samplePixelsAroundPoints(canvas, ctx, [landmarks[LM.CHIN_CREASE]], 3);
        if (samples.length > 0) {
            const variance = localVariance(samples);
            if (variance > 600 && tip === "Normală") tip = "Cu gropiță";
        }
    }
    return { tip, raportBarbie: chinRatio.toFixed(2) };
}

// ---------- Nas ----------
function classifyNose(landmarks, profileLandmarks) {
    if (profileLandmarks && profileLandmarks.length >= 468) {
        const bridgeTop = profileLandmarks[LM.NOSE_BRIDGE_TOP];
        const bridgeMid = profileLandmarks[LM.NOSE_BRIDGE_MID];
        const bridgeBot = profileLandmarks[LM.NOSE_BRIDGE_BOTTOM];
        const noseTip = profileLandmarks[LM.NOSE_TIP];
        if (bridgeTop && bridgeMid && bridgeBot && noseTip) {
            const expectedY = (bridgeTop.y + bridgeBot.y) / 2;
            const actualY = bridgeMid.y;
            const deviation = expectedY - actualY;
            const dx = noseTip.x - bridgeBot.x, dy = noseTip.y - bridgeBot.y;
            const side = detectProfileSide(profileLandmarks);
            const rawAngle = Math.atan2(dy, dx) * (180 / Math.PI);
            const angle = side === "left" ? -rawAngle : rawAngle;
            let tip;
            if (deviation > 0.02) tip = "Convex";
            else if (deviation < -0.02) tip = "Concav";
            else if (Math.abs(angle) < 20) tip = "Rectiliniu";
            else if (angle < -25) tip = "Acvilin";
            else tip = "Drept";
            return { tip, sursaAnaliza: "profil", precizieRedusa: false };
        }
    }
    const noseW = distance(landmarks[LM.RIGHT_NOSTRIL], landmarks[LM.LEFT_NOSTRIL]);
    const noseTip = landmarks[LM.NOSE_TIP], noseBridge = landmarks[LM.NOSE_BRIDGE_TOP];
    const noseLen = distance(noseTip, noseBridge);
    let tip;
    if (noseLen < 0.12) tip = "Concav";
    else if (noseW > 0.09) tip = "Convex";
    else tip = "Drept";
    return { tip, sursaAnaliza: "frontal", precizieRedusa: true };
}

// ---------- Păr ----------
function classifyHair(landmarks, canvas, ctx, normalizer) {
    const hairline = landmarks[LM.HAIRLINE_CENTER];
    const hairlineR = landmarks[LM.HAIRLINE_RIGHT];
    const hairlineL = landmarks[LM.HAIRLINE_LEFT];
    const faceHeight = distance(landmarks[LM.HAIRLINE_CENTER], landmarks[LM.CHIN]);
    const samplePoints = [];
    const leftX = hairlineL.x;
    const rightX = hairlineR.x;
    const topY = clamp(hairline.y - 0.25 * faceHeight, 0.02, 0.85);
    const bottomY = clamp(hairline.y - 0.08, 0.02, 0.85);
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 5; j++) {
            const x = leftX + (rightX - leftX) * (i / 6);
            const y = bottomY + (topY - bottomY) * (j / 4);
            samplePoints.push({ x: clamp(x, 0.05, 0.95), y: clamp(y, 0.02, 0.85) });
        }
    }
    const samples = samplePixelsAroundPoints(canvas, ctx, samplePoints, 2);
    let culoare = "Nedeterminată", textura = "Nedeterminată", calvitie = "Fără calviție";
    if (samples.length > 0) {
        const normalized = samples.map(normalizer);
        const med = medianColor(normalized);
        const hsl = rgbToHsl(med.r, med.g, med.b);
        culoare = classifyHairColor(hsl.l);

        const lums = normalized.map(luminance);
        const variance = localVariance(normalized);
        if (variance > 2500) textura = "Creț";
        else if (variance > 1200) textura = "Ondulat";
        else textura = "Drept";

        const skinSamples = samplePixelsAroundPoints(canvas, ctx, [landmarks[LM.FOREHEAD_SKIN]], 2);
        if (skinSamples.length > 0) {
            const skinMed = medianColor(skinSamples.map(normalizer));
            const skinLum = luminance(skinMed);
            const hairLum = luminance(med);
            const diff = Math.abs(skinLum - hairLum);
            if (diff < 25) calvitie = "Chelie totală";
            else if (diff < 55) calvitie = "Calviție frontală";
        }
    }
    return { culoare, textura, calvitie };
}

// ---------- Sprâncene ----------
function classifyEyebrows(landmarks, canvas, ctx) {
    const results = [];
    const rightBrowPts = [LM.RIGHT_BROW_OUTER, LM.RIGHT_BROW_TOP, LM.RIGHT_BROW_INNER];
    const leftBrowPts = [LM.LEFT_BROW_OUTER, LM.LEFT_BROW_TOP, LM.LEFT_BROW_INNER];
    let maxCurvature = 0;
    for (const pts of [rightBrowPts, leftBrowPts]) {
        const outer = landmarks[pts[0]], top = landmarks[pts[1]], inner = landmarks[pts[2]];
        const avgY = (outer.y + inner.y) / 2;
        const curvature = Math.abs(top.y - avgY);
        maxCurvature = Math.max(maxCurvature, curvature);
    }
    results.push(maxCurvature > 0.018 ? "Arcuite" : "Drepte");

    const browPoints = [LM.RIGHT_BROW_TOP, LM.LEFT_BROW_TOP, LM.RIGHT_BROW_INNER, LM.RIGHT_BROW_OUTER, LM.LEFT_BROW_INNER, LM.LEFT_BROW_OUTER].map(idx => landmarks[idx]);
    const samples = samplePixelsAroundPoints(canvas, ctx, browPoints, 3);
    if (samples.length > 0) {
        const darkCount = samples.filter(c => luminance(c) < 100).length;
        const density = darkCount / samples.length;
        if (density < 0.05) results.push("Absente");
        else if (density > 0.55) results.push("Dese");
        else if (density < 0.25) results.push("Rare");
        else results.push("Stufoase");
    } else {
        results.push("Absente");
    }
    return results;
}

// ---------- Barbă și mustață ----------
function classifyBeardAndMustache(landmarks, canvas, ctx, normalizer) {
    const skinSamples = samplePixelsAroundPoints(canvas, ctx, [landmarks[LM.RIGHT_CHEEK_SKIN]], 2);
    const skinLum = skinSamples.length > 0 ? skinSamples.map(c => luminance(c)).reduce((a, b) => a + b, 0) / skinSamples.length : 150;

    const chin = landmarks[LM.CHIN], rightJaw = landmarks[LM.RIGHT_JAW], leftJaw = landmarks[LM.LEFT_JAW];
    const beardPoints = [];
    for (let t = 0; t <= 1; t += 0.2) {
        const x = rightJaw.x + (leftJaw.x - rightJaw.x) * t;
        const y = rightJaw.y + (chin.y - rightJaw.y) * t * 1.5;
        beardPoints.push({ x: clamp(x, 0.05, 0.95), y: clamp(y, 0.1, 0.95) });
    }
    const beardSamples = samplePixelsAroundPoints(canvas, ctx, beardPoints, 2);
    const beardDarkRatio = beardSamples.length > 0 ? beardSamples.filter(c => luminance(c) < skinLum - 40).length / beardSamples.length : 0;
    const beardVariance = localVariance(beardSamples);

    let barba = "Fără barbă";
    if (beardDarkRatio > 0.55 || (beardDarkRatio > 0.25 && beardVariance > 300)) barba = "Barbă completă";
    else if (beardDarkRatio > 0.35 || (beardDarkRatio > 0.15 && beardVariance > 200)) barba = "Barbă medie";
    else if (beardDarkRatio > 0.18 || beardVariance > 120) barba = "Barbă scurtă";

    const mouthTop = landmarks[LM.MOUTH_TOP];
    const mustachePoints = [];
    for (let t = 0; t <= 1; t += 0.15) {
        const x = landmarks[LM.MOUTH_RIGHT].x + (landmarks[LM.MOUTH_LEFT].x - landmarks[LM.MOUTH_RIGHT].x) * t;
        const y = mouthTop.y - 0.015;
        mustachePoints.push({ x: clamp(x, 0.05, 0.95), y: clamp(y, 0.05, 0.95) });
    }
    const mustacheSamples = samplePixelsAroundPoints(canvas, ctx, mustachePoints, 2);
    const mustacheDarkRatio = mustacheSamples.length > 0 ? mustacheSamples.filter(c => luminance(c) < skinLum - 30).length / mustacheSamples.length : 0;
    const mustacheVariance = localVariance(mustacheSamples);

    let mustata = "Fără mustață";
    if (mustacheDarkRatio > 0.5 || (mustacheDarkRatio > 0.2 && mustacheVariance > 150)) mustata = "Groasă";
    else if (mustacheDarkRatio > 0.3 || mustacheVariance > 100) mustata = "Subțire";

    return { barba, mustata };
}

// ============================================================
// DETECȚIA URECHEI – CU FALLBACK GEOMETRIC
// ============================================================
function geometricEarEstimate(profileLandmarks, faceHeight) {
    const side = detectProfileSide(profileLandmarks);
    const { temple, jaw, cheekbone } = getSideLandmarks(profileLandmarks, side);
    if (!temple || !jaw || !cheekbone) {
        return { forma: "Nedeterminată", marime: "Nedeterminată", lob: "Nedeterminat" };
    }

    const height = Math.abs(temple.y - jaw.y) * 2.2;
    let width = Math.abs(temple.x - jaw.x) * 1.8;
    if (width < 0.02) {
        width = faceHeight * 0.15;
    }
    if (height < 0.01 || width < 0.01) {
        return { forma: "Nedeterminată", marime: "Nedeterminată", lob: "Nedeterminat" };
    }

    const aspectRatio = height / width;
    const relativeHeight = height / faceHeight;

    let forma;
    if (aspectRatio > 1.8) forma = "Dreptunghiulară";
    else if (aspectRatio > 1.4) forma = "Ovală";
    else if (aspectRatio > 1.0) forma = "Rotundă";
    else if (aspectRatio > 0.7) forma = "Triunghiulară";
    else forma = "Neregulată";

    let marime;
    if (relativeHeight < 0.22) marime = "Mici";
    else if (relativeHeight > 0.32) marime = "Mari";
    else marime = "Medii";

    return { forma, marime, lob: "Nedeterminat" };
}

async function detectEars(profileImageData, profileLandmarks, faceWidth, faceHeight) {
    // Dacă OpenCV este disponibil, încercăm detecția pe contururi
    if (window.cv && opencvReady) {
        let src = null, gray = null, edges = null, contours = null, hierarchy = null;
        const extractedMats = [];
        try {
            const canvas = profileImageData.canvas;
            const side = detectProfileSide(profileLandmarks);
            const { temple, jaw, cheekbone } = getSideLandmarks(profileLandmarks, side);
            if (temple && jaw && cheekbone) {
                const centerX = (temple.x + jaw.x) / 2;
                const centerY = (temple.y + cheekbone.y) / 2;
                const earWidth = Math.abs(temple.x - jaw.x) * 1.8;
                const earHeight = Math.abs(temple.y - jaw.y) * 2.2;
                const left = clamp(centerX - earWidth / 2, 0.05, 0.95);
                const right = clamp(centerX + earWidth / 2, 0.05, 0.95);
                const top = clamp(centerY - earHeight / 2, 0.02, 0.85);
                const bottom = clamp(centerY + earHeight / 2, 0.1, 0.95);
                const pxLeft = Math.round(left * canvas.width);
                const pxRight = Math.round(right * canvas.width);
                const pxTop = Math.round(top * canvas.height);
                const pxBottom = Math.round(bottom * canvas.height);
                const roiWidth = pxRight - pxLeft;
                const roiHeight = pxBottom - pxTop;
                if (roiWidth >= 20 && roiHeight >= 20) {
                    const roiCanvas = document.createElement("canvas");
                    roiCanvas.width = roiWidth;
                    roiCanvas.height = roiHeight;
                    const roiCtx = roiCanvas.getContext("2d");
                    roiCtx.drawImage(canvas, pxLeft, pxTop, roiWidth, roiHeight, 0, 0, roiWidth, roiHeight);

                    src = cv.imread(roiCanvas);
                    gray = new cv.Mat();
                    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
                    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
                    edges = new cv.Mat();
                    cv.Canny(gray, edges, 50, 150);
                    contours = new cv.MatVector();
                    hierarchy = new cv.Mat();
                    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                    let bestContour = null;
                    let maxArea = 0;
                    for (let i = 0; i < contours.size(); i++) {
                        const contour = contours.get(i);
                        extractedMats.push(contour);
                        const area = cv.contourArea(contour);
                        if (area > maxArea && area > 100) {
                            maxArea = area;
                            bestContour = contour;
                        }
                    }

                    if (bestContour) {
                        const rect = cv.boundingRect(bestContour);
                        const aspectRatio = rect.height / rect.width;
                        const relativeHeight = rect.height / faceHeight;
                        let forma, marime;
                        if (aspectRatio > 1.8) forma = "Dreptunghiulară";
                        else if (aspectRatio > 1.4) forma = "Ovală";
                        else if (aspectRatio > 1.0) forma = "Rotundă";
                        else if (aspectRatio > 0.7) forma = "Triunghiulară";
                        else forma = "Neregulată";
                        if (relativeHeight < 0.22) marime = "Mici";
                        else if (relativeHeight > 0.32) marime = "Mari";
                        else marime = "Medii";
                        // Lobul rămâne nedeterminat; nu putem fi siguri
                        return { forma, marime, lob: "Nedeterminat" };
                    }
                }
            }
        } catch (err) {
            console.warn("Eroare la detecția OpenCV a urechii, folosim fallback geometric:", err);
        } finally {
            for (const mat of extractedMats) mat.delete();
            if (src) src.delete();
            if (gray) gray.delete();
            if (edges) edges.delete();
            if (contours) contours.delete();
            if (hierarchy) hierarchy.delete();
        }
    }
    // Fallback geometric dacă OpenCV nu este disponibil sau conturul nu a fost găsit
    return geometricEarEstimate(profileLandmarks, faceHeight);
}

// ============================================================
// FLUX PRINCIPAL DE ANALIZĂ
// ============================================================
async function runAnalysis() {
    const statusEl = document.getElementById("status");
    statusEl.className = "status info";
    statusEl.textContent = "Se procesează imaginile...";
    const btn = document.getElementById("btn-analyze");
    btn.disabled = true;
    document.getElementById("analyze-spinner").style.display = "inline-block";
    document.getElementById("analyze-text").textContent = "Se analizează...";

    currentResults = null;
    document.getElementById("results-section").classList.remove("visible");

    let frontalProc = null, profilProc = null;

    try {
        try { await loadOpenCV(); } catch (cvError) {
            console.warn("OpenCV.js nu a putut fi încărcat. Detecția urechii va folosi doar fallback geometric.");
        }

        if (!frontalFile) throw new Error("Încarcă poza din față.");
        frontalProc = await processImage(frontalFile);
        const frontalLandmarks = await extractLandmarks(frontalProc);
        if (!frontalLandmarks) {
            statusEl.className = "status error";
            statusEl.textContent = "Nu s-au putut detecta landmark-urile faciale în poza din față. Verifică încadrarea și claritatea.";
            purgeImageData(frontalProc);
            return;
        }

        let profilLandmarks = null;
        if (profilFile) {
            try {
                profilProc = await processImage(profilFile);
                profilLandmarks = await extractLandmarks(profilProc);
                if (!profilLandmarks) console.warn("Nu s-au detectat landmark-uri în poza de profil. Folosim doar analiza frontală.");
            } catch (profilErr) { console.warn("Eroare la procesarea pozei de profil:", profilErr); }
        }

        const faceWidth = distance(frontalLandmarks[LM.FACE_RIGHT_TEMPLE], frontalLandmarks[LM.FACE_LEFT_TEMPLE]);
        const faceHeight = distance(frontalLandmarks[LM.HAIRLINE_CENTER], frontalLandmarks[LM.CHIN]);

        // Construim normalizatorul de culoare pe baza pielii frunții
        const normalizer = createColorNormalizer(frontalProc.canvas, frontalProc.ctx, frontalLandmarks);

        // Detecția urechilor (doar dacă avem profil)
        let urechi = { forma: "Nedeterminată", marime: "Nedeterminată", lob: "Nedeterminat" };
        if (profilProc && profilLandmarks) {
            try {
                urechi = await detectEars(profilProc, profilLandmarks, faceWidth, faceHeight);
            } catch (earErr) { console.warn("Eroare la detecția urechii:", earErr); }
        }

        // Clasificări
        const barbaMustata = classifyBeardAndMustache(frontalLandmarks, frontalProc.canvas, frontalProc.ctx, normalizer);
        const nas = classifyNose(frontalLandmarks, profilLandmarks);
        const par = classifyHair(frontalLandmarks, frontalProc.canvas, frontalProc.ctx, normalizer);
        const ochi = classifyEyes(frontalLandmarks, frontalProc.canvas, frontalProc.ctx, faceWidth, normalizer);

        const results = {
            frunte: classifyForehead(frontalLandmarks, faceWidth),
            nas: nas,
            ochi: ochi,
            gura: classifyMouth(frontalLandmarks, faceWidth),
            barbie: classifyChin(frontalLandmarks, faceWidth, frontalProc.canvas, frontalProc.ctx),
            tipFata: classifyFaceType(frontalLandmarks),
            par: par,
            sprancene: classifyEyebrows(frontalLandmarks, frontalProc.canvas, frontalProc.ctx),
            barba: barbaMustata.barba,
            mustata: barbaMustata.mustata,
            urechi: urechi,
            semneParticulare: ""
        };

        currentResults = results;
        renderResults(results);
        document.getElementById("results-section").classList.add("visible");
        statusEl.className = "status success";
        statusEl.textContent = "Analiza completă! Rezultatele detectate sunt afișate mai jos.";
    } catch (err) {
        console.error("Eroare în analiză:", err);
        statusEl.className = "status error";
        statusEl.textContent = "Eroare: " + err.message;
    } finally {
        purgeImageData(frontalProc);
        purgeImageData(profilProc);
        console.log("Datele de imagine au fost eliminate din canvas/memorie.");
        btn.disabled = false;
        document.getElementById("analyze-spinner").style.display = "none";
        document.getElementById("analyze-text").textContent = "Analizează fețele";
        checkAnalyzeButton();
    }
}

// ============================================================
// RENDERIZARE REZULTATE – AFIȘARE SIMPLĂ + INPUT MANUAL SEMNE
// ============================================================
function renderResults(results) {
    const grid = document.getElementById("results-grid");
    grid.innerHTML = "";

    grid.appendChild(createCard("Fruntea", [
        makeTextValue("Tipul frunții", results.frunte?.tip || "Mijlocie")
    ]));

    const nasFields = [ makeTextValue("Tipul nasului", results.nas?.tip || "Drept") ];
    if (results.nas?.precizieRedusa) {
        const warn = document.createElement("p");
        warn.style.cssText = "font-size:0.75rem;color:#f0a020;margin-top:4px;";
        warn.textContent = "Analizat doar din poza frontală — fără poza din profil, precizia pentru forma nasului este redusă.";
        nasFields.push(warn);
    }
    grid.appendChild(createCard("Nasul", nasFields));

    grid.appendChild(createCard("Ochii", [
        makeTextValue("Culoarea ochilor", results.ochi?.culoare || "Nedeterminată"),
        makeTextValue("Mărimea ochilor", results.ochi?.marime || "Nedeterminată")
    ]));

    grid.appendChild(createCard("Gura", [
        makeTextValue("Colțurile gurii", results.gura?.colturi || "Nedeterminată"),
        makeTextValue("Mărimea gurii", results.gura?.marime || "Nedeterminată")
    ]));

    grid.appendChild(createCard("Bărbia", [
        makeTextValue("Tipul bărbiei", results.barbie?.tip || "Nedeterminată")
    ]));

    grid.appendChild(createCard("Tipul feței", [
        makeTextValue("Forma feței", results.tipFata?.tip || "Nedeterminată")
    ]));

    grid.appendChild(createCard("Părul", [
        makeTextValue("Culoarea părului", results.par?.culoare || "Nedeterminată"),
        makeTextValue("Textura părului", results.par?.textura || "Nedeterminată"),
        makeTextValue("Calviția", results.par?.calvitie || "Nedeterminată")
    ]));

    grid.appendChild(createCard("Sprâncenele", [
        makeTextValue("Caracteristici", results.sprancene ? results.sprancene.join(", ") : "Absente")
    ]));

    grid.appendChild(createCard("Barba", [
        makeTextValue("Tipul bărbii", results.barba || "Fără barbă")
    ]));

    grid.appendChild(createCard("Mustața", [
        makeTextValue("Tipul mustății", results.mustata || "Fără mustață")
    ]));

    // Card urechi
    grid.appendChild(createCard("Urechile", [
        makeTextValue("Forma urechii", results.urechi?.forma || "Nedeterminată"),
        makeTextValue("Mărimea urechii", results.urechi?.marime || "Nedeterminată"),
        makeTextValue("Lobul urechii", results.urechi?.lob || "Nedeterminat")
    ]));

    // Card semne particulare – cu input text manual
    const semneContainer = document.createElement("div");
    semneContainer.className = "field";
    const semneLabel = document.createElement("label");
    semneLabel.className = "field-label";
    semneLabel.textContent = "Tatuaje, cicatrici etc.";
    const semneInput = document.createElement("input");
    semneInput.type = "text";
    semneInput.id = "semne-text";
    semneInput.placeholder = "Introduceți manual observații...";
    semneInput.value = results.semneParticulare || "";
    semneContainer.appendChild(semneLabel);
    semneContainer.appendChild(semneInput);
    grid.appendChild(createCard("Semne particulare", [semneContainer]));

    const infoCard = document.createElement("div");
    infoCard.className = "result-card";
    infoCard.style.borderColor = "rgba(255,255,255,0.15)";
    infoCard.innerHTML = `
        <div class="card-title">Fiabilitate</div>
        <p style="font-size:0.8rem;color:var(--text-secondary);">
            Categoriile geometrice (tip față, gură, frunte, sprâncene) au fiabilitate ridicată.
            Culoarea ochilor/părului este aproximativă (sampling de culoare cu normalizare).
            Tipul nasului este mult mai precis cu poza de profil inclusă. Barba/mustața sunt orientative.
            Detecția urechilor este experimentală și poate fi imprecisă.
            Verifică rezultatele înainte de salvare.
        </p>
    `;
    grid.appendChild(infoCard);
}

function createCard(title, fields) {
    const card = document.createElement("div");
    card.className = "result-card";
    const titleEl = document.createElement("div");
    titleEl.className = "card-title";
    titleEl.textContent = title;
    card.appendChild(titleEl);
    fields.forEach(f => card.appendChild(f));
    return card;
}

function makeTextValue(labelText, value) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = labelText;
    const valueEl = document.createElement("span");
    valueEl.className = "field-value";
    valueEl.textContent = value;
    wrapper.appendChild(label);
    wrapper.appendChild(valueEl);
    return wrapper;
}

// ============================================================
// COLECTARE REZULTATE (inclusiv input-ul manual de semne)
// ============================================================
function collectResultsFromUI() {
    const semne = document.getElementById("semne-text")?.value || "";
    return {
        ...currentResults,
        semneParticulare: semne,
        dataAnaliza: new Date().toISOString()
    };
}

// ============================================================
// SALVARE / EXPORT
// ============================================================
function saveResults() {
    if (!currentResults) {
        alert("Nu există rezultate de salvat. Rulează mai întâi analiza.");
        return;
    }
    const data = collectResultsFromUI();
    const key = "semnalmente:" + Date.now();
    try {
        localStorage.setItem(key, JSON.stringify(data));
        alert("Fișa a fost salvată în localStorage sub cheia: " + key);
        renderSavedList();
    } catch (err) {
        alert("Eroare la salvare: " + err.message);
    }
}

function exportResults() {
    if (!currentResults) {
        alert("Nu există rezultate de exportat.");
        return;
    }
    const data = collectResultsFromUI();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "semnalmente_" + Date.now() + ".json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================
// LISTA FIȘELOR SALVATE
// ============================================================
function renderSavedList() {
    const container = document.getElementById("saved-items");
    container.innerHTML = "";
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith("semnalmente:")) keys.push(key);
    }
    keys.sort().reverse();
    if (keys.length === 0) {
        container.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem;">Nicio fișă salvată.</p>';
        return;
    }
    keys.forEach(key => {
        const raw = localStorage.getItem(key);
        let parsed;
        try { parsed = JSON.parse(raw); } catch { parsed = { dataAnaliza: "necunoscută" }; }
        const item = document.createElement("div");
        item.className = "saved-item";
        const date = parsed.dataAnaliza ? new Date(parsed.dataAnaliza).toLocaleString("ro-RO") : "dată necunoscută";
        item.innerHTML = `
            <span style="color:var(--text);"><strong>${key}</strong><br>
            <span style="color:var(--text-secondary);font-size:0.75rem;">${date}</span></span>
        `;
        const actions = document.createElement("div");
        actions.className = "saved-actions";
        const loadBtn = document.createElement("button");
        loadBtn.textContent = "Încarcă";
        loadBtn.addEventListener("click", () => loadSavedData(parsed));
        const delBtn = document.createElement("button");
        delBtn.textContent = "Șterge";
        delBtn.className = "delete";
        delBtn.addEventListener("click", () => {
            if (confirm("Șterge această fișă?")) {
                localStorage.removeItem(key);
                renderSavedList();
            }
        });
        actions.appendChild(loadBtn);
        actions.appendChild(delBtn);
        item.appendChild(actions);
        container.appendChild(item);
    });
}

function loadSavedData(data) {
    // Normalizări minime
    if (!data.frunte) data.frunte = { tip: "Mijlocie" };
    if (!data.nas) data.nas = { tip: "Drept", precizieRedusa: false };
    if (!data.ochi) data.ochi = { culoare: "Nedeterminată", marime: "Mijlocii" };
    if (!data.gura) data.gura = { colturi: "Liniară", marime: "Mijlocie" };
    if (!data.barbie) data.barbie = { tip: "Normală" };
    if (!data.tipFata) data.tipFata = { tip: "Ovală" };
    if (!data.par) data.par = { culoare: "Nedeterminată", textura: "Nedeterminată", calvitie: "Fără calviție" };
    if (!data.sprancene) data.sprancene = [];
    if (!data.barba) data.barba = "Fără barbă";
    if (!data.mustata) data.mustata = "Fără mustață";
    if (!data.urechi) data.urechi = { forma: "Nedeterminată", marime: "Nedeterminată", lob: "Nedeterminat" };
    if (!data.semneParticulare) data.semneParticulare = "";

    currentResults = data;
    renderResults(data);
    document.getElementById("results-section").classList.add("visible");
    document.getElementById("status").className = "status info";
    document.getElementById("status").textContent = "Fișă încărcată. Rezultatele sunt afișate.";
}

// ============================================================
// RESETARE
// ============================================================
function resetAll() {
    if (confirm("Resetezi toate datele? Se vor pierde rezultatele curente.")) {
        currentResults = null;
        frontalFile = null;
        profilFile = null;
        document.getElementById("results-section").classList.remove("visible");
        document.getElementById("status").className = "status";
        document.getElementById("status").textContent = "";
        ["frontal", "profil"].forEach(prefix => {
            const preview = document.getElementById(`preview-${prefix}`);
            preview.src = "";
            preview.classList.remove("visible");
            document.getElementById(`remove-${prefix}`).classList.remove("visible");
            document.getElementById(`file-${prefix}`).value = "";
        });
        objectUrls.forEach(url => URL.revokeObjectURL(url));
        objectUrls = [];
        checkAnalyzeButton();
        renderSavedList();
    }
}

// ============================================================
// INIȚIALIZARE
// ============================================================
async function initApp() {
    setupUploads();
    checkAnalyzeButton();
    renderSavedList();
    document.getElementById("btn-analyze").addEventListener("click", runAnalysis);
    document.getElementById("btn-save").addEventListener("click", saveResults);
    document.getElementById("btn-export").addEventListener("click", exportResults);
    document.getElementById("btn-reset").addEventListener("click", resetAll);
    try {
        await initFaceLandmarker();
        console.log("Aplicație pregătită. Poți încărca imagini.");
    } catch (err) {
        console.error("Eroare la inițializarea MediaPipe:", err);
        document.getElementById("status").className = "status error";
        document.getElementById("status").textContent =
            "Eroare la încărcarea modelului MediaPipe. Verifică conexiunea la internet (CDN-urile trebuie să fie accesibile).";
    }
}

initApp();