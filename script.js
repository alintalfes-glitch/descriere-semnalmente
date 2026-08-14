// ============================================================
// APLICAȚIE DE ANALIZĂ FACIALĂ – script.js
// Modul ES, 100% client-side.
// Folosește MediaPipe Face Landmarker pentru extragerea
// a 468+ landmark-uri faciale.
// ============================================================

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// ============================================================
// CONFIGURARE
// ============================================================
const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_PATH =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// ============================================================
// STARE GLOBALĂ
// ============================================================
let faceLandmarker = null;          // instanța MediaPipe
let frontalImageData = null;        // { image, canvas, ctx }
let profilImageData = null;         // { image, canvas, ctx }
let currentResults = null;          // rezultatele curente (obiect)
let objectUrls = [];                // URL-uri create pentru curățenie
let frontalFile = null;             // fișierul frontal
let profilFile = null;              // fișierul profil

// ============================================================
// OPȚIUNI PENTRU DROPDOWN-URI ȘI CHECKBOX-URI
// ============================================================
const OPTIONS = {
    frunte: ["Îngustă", "Lată", "Proeminentă", "Înaltă", "Mijlocie", "Scundă", "Nedeterminată"],
    nas: [
        "Acvilin", "Concav", "Convex", "Coroiat", "Drept",
        "Frânt jos", "Frânt sus", "Ondulat", "Rectiliniu", "Nedeterminat"
    ],
    ochiCuloare: [
        "Albaștri", "Căprui", "Cenușii", "Negri", "Verzi",
        "Ceacâr (heterocromie)", "Nedeterminată"
    ],
    ochiMarime: ["Mici", "Mijlocii", "Mari", "Nedeterminată"],
    guraColturi: ["Colțuri coborâte", "Colțuri ridicate", "Liniară", "Nedeterminată"],
    guraMarime: ["Mare", "Mijlocie", "Mică", "Nedeterminată"],
    barbie: [
        "Alungită", "Ascuțită", "Bilobată", "Cu gropiță", "Dublă",
        "Îngropată", "Normală", "Plată", "Proeminentă", "Nedeterminată"
    ],
    tipFata: [
        "Dreptunghiulară", "Ascuțită", "Ovală", "Romboidă",
        "Rotundă", "Pătrată", "Triunghiulară", "Nedeterminată"
    ],
    parCuloare: ["Negru", "Șaten", "Blond", "Roșcat", "Cărunt", "Nedeterminată"],
    parTextura: ["Ondulat", "Drept", "Creț", "Nedeterminată"],
    calvitie: ["Fără calviție", "Calviție frontală", "Calviție parietală", "Chelie totală", "Nedeterminată"],
    sprancene: ["Absente", "Arcuite", "Dese", "Drepte", "Pensate", "Rare", "Stufoase"],
    barba: [
        "Fără barbă", "Barbă scurtă", "Barbă medie", "Barbă lungă",
        "Barbă tip cioc", "Barbă completă", "Nedeterminată"
    ],
    mustata: [
        "Fără mustață", "Subțire", "Groasă", "Dreaptă",
        "Cu colțurile ridicate", "Cu colțurile coborâte", "Nedeterminată"
    ],
    urechiForma: ["Ovală", "Rotundă", "Dreptunghiulară", "Triunghiulară", "Neregulată", "Nedeterminată"],
    urechiMarime: ["Mici", "Medii", "Mari", "Nedeterminată"],
    urechiLob: ["Atașat", "Liber", "Nedeterminat"],
};

// ============================================================
// INDICI LANDMARK-URI MEDIAPIPE FACE MESH (esențiali)
// ============================================================
const LM = {
    // Conturul feței
    FACE_RIGHT_TEMPLE: 234,
    FACE_LEFT_TEMPLE: 454,
    RIGHT_JAW: 58,
    LEFT_JAW: 288,
    CHIN: 152,
    // Frunte / păr
    HAIRLINE_CENTER: 10,
    HAIRLINE_RIGHT: 67,
    HAIRLINE_LEFT: 297,
    FOREHEAD_CENTER: 151,
    // Pomeți
    RIGHT_CHEEKBONE: 50,
    LEFT_CHEEKBONE: 280,
    // Ochii
    RIGHT_EYE_OUTER: 33,
    RIGHT_EYE_INNER: 133,
    LEFT_EYE_INNER: 362,
    LEFT_EYE_OUTER: 263,
    RIGHT_IRIS_CENTER: 468,
    LEFT_IRIS_CENTER: 473,
    // Sprâncene
    RIGHT_BROW_OUTER: 70,
    RIGHT_BROW_INNER: 107,
    RIGHT_BROW_TOP: 65,
    LEFT_BROW_OUTER: 300,
    LEFT_BROW_INNER: 336,
    LEFT_BROW_TOP: 295,
    // Nasul
    NOSE_TIP: 1,
    NOSE_BRIDGE_TOP: 6,
    NOSE_BRIDGE_MID: 168,
    NOSE_BRIDGE_BOTTOM: 2,
    RIGHT_NOSTRIL: 45,
    LEFT_NOSTRIL: 275,
    // Gura
    MOUTH_RIGHT: 61,
    MOUTH_LEFT: 291,
    MOUTH_TOP: 13,
    MOUTH_BOTTOM: 14,
    // Bărbie
    CHIN_RIGHT: 201,
    CHIN_LEFT: 200,
    CHIN_CREASE: 202,
    // Obraji (referință piele)
    RIGHT_CHEEK_SKIN: 116,
    LEFT_CHEEK_SKIN: 345,
    // Frunte piele (referință)
    FOREHEAD_SKIN: 8,
    // Urechi
    RIGHT_EAR_TOP: 93,
    RIGHT_EAR_BOTTOM: 132,
    RIGHT_EAR_LOBE: 136,
    LEFT_EAR_TOP: 323,
    LEFT_EAR_BOTTOM: 361,
    LEFT_EAR_LOBE: 365,
};

// ============================================================
// FUNCȚII UTILITARE
// ============================================================
function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function averageColor(colors) {
    if (!colors || colors.length === 0) return { r: 0, g: 0, b: 0 };
    const sum = colors.reduce(
        (acc, c) => {
            acc.r += c.r;
            acc.g += c.g;
            acc.b += c.b;
            return acc;
        },
        { r: 0, g: 0, b: 0 }
    );
    return {
        r: sum.r / colors.length,
        g: sum.g / colors.length,
        b: sum.b / colors.length,
    };
}

function rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h,
        s,
        l = (max + min) / 2;

    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r:
                h = (g - b) / d + (g < b ? 6 : 0);
                break;
            case g:
                h = (b - r) / d + 2;
                break;
            case b:
                h = (r - g) / d + 4;
                break;
        }
        h /= 6;
    }
    return { h: h * 360, s: s * 100, l: l * 100 };
}

function classifyColor(r, g, b) {
    // Clasificare simplă a culorii pe baza HSL
    const { h, s, l } = rgbToHsl(r, g, b);

    if (l < 15) return "Negru";
    if (l > 80 && s < 25) return "Cărunt";
    if (s < 18) {
        if (l > 65) return "Blond";
        if (l > 40) return "Cărunt";
        return "Negru";
    }
    if (h < 20) return "Roșcat";
    if (h < 45) return "Blond";
    if (h < 80) return "Șaten";
    if (h < 170) return "Șaten";
    if (h < 260) return "Negru";
    return "Negru";
}

// ============================================================
// INIȚIALIZARE MEDIAPIPE FACE LANDMARKER
// ============================================================
async function initFaceLandmarker() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(WASM_PATH);
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: MODEL_URL,
                delegate: "GPU",
            },
            runningMode: "IMAGE",
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
        });
        console.log("✅ FaceLandmarker inițializat cu succes (GPU)");
    } catch (gpuError) {
        console.warn("⚠️ GPU delegate a eșuat, încerc CPU...", gpuError);
        try {
            const filesetResolver = await FilesetResolver.forVisionTasks(WASM_PATH);
            faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: MODEL_URL,
                    delegate: "CPU",
                },
                runningMode: "IMAGE",
                numFaces: 1,
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false,
            });
            console.log("✅ FaceLandmarker inițializat cu CPU");
        } catch (cpuError) {
            console.error("❌ Eroare la inițializarea FaceLandmarker:", cpuError);
            throw cpuError;
        }
    }
}

// ============================================================
// GESTIONARE UPLOAD IMAGINI
// ============================================================
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
            objectUrls.push(url);
            preview.src = url;
            preview.classList.add("visible");
            removeBtn.classList.add("visible");
            callback(url, file);
        }
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        preview.src = "";
        preview.classList.remove("visible");
        removeBtn.classList.remove("visible");
        if (preview.src) {
            const url = preview.src;
            objectUrls = objectUrls.filter((u) => u !== url);
            URL.revokeObjectURL(url);
        }
        callback(null, null);
    });

    // Drag & drop
    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            fileInput.files = e.dataTransfer.files;
            const url = URL.createObjectURL(file);
            objectUrls.push(url);
            preview.src = url;
            preview.classList.add("visible");
            removeBtn.classList.add("visible");
            callback(url, file);
        }
    });
}

function setupUploads() {
    setupUploadZone(
        "drop-frontal",
        "file-frontal",
        "preview-frontal",
        "remove-frontal",
        (url, file) => {
            frontalFile = file;
            checkAnalyzeButton();
        }
    );
    setupUploadZone(
        "drop-profil",
        "file-profil",
        "preview-profil",
        "remove-profil",
        (url, file) => {
            profilFile = file;
            checkAnalyzeButton();
        }
    );
}

function checkAnalyzeButton() {
    const btn = document.getElementById("btn-analyze");
    btn.disabled = !frontalFile; // profilul este opțional
    const text = document.getElementById("analyze-text");
    text.textContent = frontalFile ? "🔬 Analizează fețele" : "📸 Încarcă poza din față";
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
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            resolve({ image: img, canvas, ctx });
        };
        img.onerror = (err) => {
            reject(new Error("Nu s-a putut încărca imaginea."));
        };
        img.src = url;
    });
}

async function extractLandmarks(imageData) {
    if (!faceLandmarker) await initFaceLandmarker();
    try {
        const result = await faceLandmarker.detect(imageData.image);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            const landmarks = result.faceLandmarks[0];
            // Acceptăm atât 468 cât și 478 (cu iris)
            if (landmarks.length >= 468) {
                return landmarks;
            }
            console.warn("⚠️ Număr insuficient de landmark-uri:", landmarks.length);
        }
        return null;
    } catch (err) {
        console.error("Eroare la detectarea landmark-urilor:", err);
        return null;
    }
}

function clearCanvasData(imageData) {
    if (imageData && imageData.canvas) {
        const ctx = imageData.canvas.getContext("2d");
        ctx.clearRect(0, 0, imageData.canvas.width, imageData.canvas.height);
        imageData.canvas.width = 0;
        imageData.canvas.height = 0;
    }
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

    return {
        tip: result[0] || "Mijlocie",
        detalii: result.join(", "),
        raportLatime: widthRatio.toFixed(2),
        raportInaltime: heightRatio.toFixed(2),
    };
}

// ---------- Tipul feței ----------
function classifyFaceType(landmarks) {
    const foreheadW = distance(landmarks[LM.HAIRLINE_RIGHT], landmarks[LM.HAIRLINE_LEFT]);
    const cheekboneW = distance(landmarks[LM.FACE_RIGHT_TEMPLE], landmarks[LM.FACE_LEFT_TEMPLE]);
    const jawW = distance(landmarks[LM.RIGHT_JAW], landmarks[LM.LEFT_JAW]);
    const faceH = distance(landmarks[LM.HAIRLINE_CENTER], landmarks[LM.CHIN]);

    const ratio = faceH / cheekboneW;
    const jawRatio = jawW / cheekboneW;
    const foreheadRatio = foreheadW / cheekboneW;

    let tip;
    if (ratio > 1.45 && jawRatio < 0.75 && foreheadRatio > 0.85) {
        tip = "Triunghiulară";
    } else if (ratio > 1.45 && jawRatio < 0.75 && foreheadRatio < 0.80) {
        tip = "Ascuțită";
    } else if (ratio > 1.35 && foreheadRatio > 0.88 && jawRatio > 0.82) {
        tip = "Dreptunghiulară";
    } else if (
        ratio > 1.25 &&
        foreheadRatio < 0.85 &&
        jawRatio < 0.80 &&
        cheekboneW > foreheadW &&
        cheekboneW > jawW
    ) {
        tip = "Romboidă";
    } else if (ratio > 1.25) {
        tip = "Ovală";
    } else if (ratio < 1.15 && foreheadRatio > 0.88 && jawRatio > 0.85) {
        tip = "Pătrată";
    } else if (ratio < 1.25) {
        tip = "Rotundă";
    } else {
        tip = "Ovală";
    }
    return {
        tip,
        raport: ratio.toFixed(2),
        latimePometi: cheekboneW.toFixed(2),
        latimeMaxilar: jawW.toFixed(2),
    };
}

// ---------- Ochi ----------
function classifyEyes(landmarks, canvas, ctx, faceWidth) {
    const rEyeW = distance(landmarks[LM.RIGHT_EYE_OUTER], landmarks[LM.RIGHT_EYE_INNER]);
    const lEyeW = distance(landmarks[LM.LEFT_EYE_OUTER], landmarks[LM.LEFT_EYE_INNER]);
    const avgEyeW = (rEyeW + lEyeW) / 2;
    const eyeRatio = avgEyeW / faceWidth;

    let marime;
    if (eyeRatio < 0.15) marime = "Mici";
    else if (eyeRatio > 0.21) marime = "Mari";
    else marime = "Mijlocii";

    // Culoare iris – sampling din zona irisului
    let culoare = "Nedeterminată";
    const samples = [];
    const irisPoints = [LM.RIGHT_IRIS_CENTER, LM.LEFT_IRIS_CENTER];
    for (const pt of irisPoints) {
        if (pt < landmarks.length) {
            const l = landmarks[pt];
            const cx = Math.round(l.x * canvas.width);
            const cy = Math.round(l.y * canvas.height);
            // Sampling într-o zonă mică 5x5
            for (let dx = -2; dx <= 2; dx++) {
                for (let dy = -2; dy <= 2; dy++) {
                    const px = clamp(cx + dx, 0, canvas.width - 1);
                    const py = clamp(cy + dy, 0, canvas.height - 1);
                    const pixel = ctx.getImageData(px, py, 1, 1).data;
                    samples.push({ r: pixel[0], g: pixel[1], b: pixel[2] });
                }
            }
        }
    }
    if (samples.length > 0) {
        const avg = averageColor(samples);
        culoare = classifyColor(avg.r, avg.g, avg.b);
        // Mapare aproximativă
        if (culoare === "Șaten") culoare = "Căprui";
        else if (culoare === "Blond") culoare = "Albaștri";
        else if (culoare === "Roșcat") culoare = "Căprui";
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

    const cornerR = landmarks[LM.MOUTH_RIGHT];
    const cornerL = landmarks[LM.MOUTH_LEFT];
    const cornerAvgY = (cornerR.y + cornerL.y) / 2;
    const topY = landmarks[LM.MOUTH_TOP].y;

    let colturi;
    const diff = topY - cornerAvgY;
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

    // Detectare gropiță – variație de luminozitate în centrul bărbiei
    if (canvas && ctx && LM.CHIN_CREASE < landmarks.length) {
        const c = landmarks[LM.CHIN_CREASE];
        const cx = Math.round(c.x * canvas.width);
        const cy = Math.round(c.y * canvas.height);
        const samples = [];
        for (let dx = -3; dx <= 3; dx++) {
            for (let dy = -3; dy <= 3; dy++) {
                const px = clamp(cx + dx, 0, canvas.width - 1);
                const py = clamp(cy + dy, 0, canvas.height - 1);
                const pixel = ctx.getImageData(px, py, 1, 1).data;
                const lum = (pixel[0] + pixel[1] + pixel[2]) / 3;
                samples.push(lum);
            }
        }
        const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
        const variance = samples.reduce((acc, v) => acc + (v - avg) ** 2, 0) / samples.length;
        if (variance > 600 && tip === "Normală") {
            tip = "Cu gropiță";
        }
    }
    return { tip, raportBarbie: chinRatio.toFixed(2) };
}

// ---------- Nas ----------
function classifyNose(landmarks, profileLandmarks) {
    // Preferăm analiza de profil, dacă există
    if (profileLandmarks && profileLandmarks.length >= 468) {
        const bridgeTop = profileLandmarks[LM.NOSE_BRIDGE_TOP];
        const bridgeMid = profileLandmarks[LM.NOSE_BRIDGE_MID];
        const bridgeBot = profileLandmarks[LM.NOSE_BRIDGE_BOTTOM];
        const noseTip = profileLandmarks[LM.NOSE_TIP];

        if (bridgeTop && bridgeMid && bridgeBot && noseTip) {
            // Deviația punctului median față de linia top-bottom
            const expectedY = (bridgeTop.y + bridgeBot.y) / 2;
            const actualY = bridgeMid.y;
            const deviation = expectedY - actualY;

            // Unghiul dintre bridgeBot și vârful nasului
            const dx = noseTip.x - bridgeBot.x;
            const dy = noseTip.y - bridgeBot.y;
            const angle = Math.atan2(dy, dx) * (180 / Math.PI);

            if (deviation > 0.02) return "Convex"; // cocoașă (acvilin)
            if (deviation < -0.02) return "Concav"; // nas în șa
            if (Math.abs(angle) < 20) return "Rectiliniu";
            if (angle < -25) return "Acvilin";
            return "Drept";
        }
    }

    // Fallback: analiză frontală
    const noseW = distance(landmarks[LM.RIGHT_NOSTRIL], landmarks[LM.LEFT_NOSTRIL]);
    const noseTip = landmarks[LM.NOSE_TIP];
    const noseBridge = landmarks[LM.NOSE_BRIDGE_TOP];
    const noseLen = distance(noseTip, noseBridge);
    if (noseLen < 0.12) return "Concav";
    if (noseW > 0.09) return "Convex";
    return "Drept";
}

// ---------- Păr ----------
function classifyHair(landmarks, canvas, ctx) {
    const samples = [];
    const hairline = landmarks[LM.HAIRLINE_CENTER];
    const hairlineR = landmarks[LM.HAIRLINE_RIGHT];
    const hairlineL = landmarks[LM.HAIRLINE_LEFT];

    const samplePoints = [
        { x: hairline.x, y: hairline.y - 0.08 },
        { x: hairlineR.x, y: hairlineR.y - 0.06 },
        { x: hairlineL.x, y: hairlineL.y - 0.06 },
        { x: hairline.x, y: hairline.y - 0.12 },
    ];

    for (const sp of samplePoints) {
        const cx = Math.round(clamp(sp.x, 0.05, 0.95) * canvas.width);
        const cy = Math.round(clamp(sp.y, 0.02, 0.85) * canvas.height);
        for (let dx = -2; dx <= 2; dx++) {
            for (let dy = -2; dy <= 2; dy++) {
                const px = clamp(cx + dx, 0, canvas.width - 1);
                const py = clamp(cy + dy, 0, canvas.height - 1);
                const pixel = ctx.getImageData(px, py, 1, 1).data;
                samples.push({ r: pixel[0], g: pixel[1], b: pixel[2] });
            }
        }
    }

    let culoare = "Nedeterminată";
    let textura = "Nedeterminată";
    let calvitie = "Fără calviție";

    if (samples.length > 0) {
        const avg = averageColor(samples);
        culoare = classifyColor(avg.r, avg.g, avg.b);

        // Varianță pentru textură
        const luminances = samples.map((c) => (c.r + c.g + c.b) / 3);
        const avgLum = luminances.reduce((a, b) => a + b, 0) / luminances.length;
        const variance = luminances.reduce((acc, v) => acc + (v - avgLum) ** 2, 0) / luminances.length;
        if (variance > 2500) textura = "Creț";
        else if (variance > 1200) textura = "Ondulat";
        else textura = "Drept";

        // Calviție: compară culoarea scalpului cu pielea frunții
        const skinL = landmarks[LM.FOREHEAD_SKIN];
        const sx = Math.round(clamp(skinL.x, 0, 1) * canvas.width);
        const sy = Math.round(clamp(skinL.y, 0, 1) * canvas.height);
        const skinPixel = ctx.getImageData(
            clamp(sx, 0, canvas.width - 1),
            clamp(sy, 0, canvas.height - 1),
            1,
            1
        ).data;
        const skinLum = (skinPixel[0] + skinPixel[1] + skinPixel[2]) / 3;

        if (Math.abs(skinLum - avgLum) < 25) {
            calvitie = "Chelie totală";
        } else if (Math.abs(skinLum - avgLum) < 55) {
            calvitie = "Calviție frontală";
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
        const outer = landmarks[pts[0]];
        const top = landmarks[pts[1]];
        const inner = landmarks[pts[2]];
        const avgY = (outer.y + inner.y) / 2;
        const curvature = Math.abs(top.y - avgY);
        maxCurvature = Math.max(maxCurvature, curvature);
    }
    if (maxCurvature > 0.018) results.push("Arcuite");
    else results.push("Drepte");

    // Densitate (pixeli închiși în zona sprâncenelor)
    let darkCount = 0;
    let totalSamples = 0;
    for (const idx of [
        LM.RIGHT_BROW_TOP,
        LM.LEFT_BROW_TOP,
        LM.RIGHT_BROW_INNER,
        LM.RIGHT_BROW_OUTER,
        LM.LEFT_BROW_INNER,
        LM.LEFT_BROW_OUTER,
    ]) {
        const l = landmarks[idx];
        const cx = Math.round(l.x * canvas.width);
        const cy = Math.round(l.y * canvas.height);
        for (let dx = -3; dx <= 3; dx++) {
            for (let dy = -3; dy <= 3; dy++) {
                const px = clamp(cx + dx, 0, canvas.width - 1);
                const py = clamp(cy + dy, 0, canvas.height - 1);
                const pixel = ctx.getImageData(px, py, 1, 1).data;
                const lum = (pixel[0] + pixel[1] + pixel[2]) / 3;
                totalSamples++;
                if (lum < 100) darkCount++;
            }
        }
    }
    const density = darkCount / totalSamples;
    if (density > 0.55) results.push("Dese");
    else if (density < 0.25) results.push("Rare");
    else results.push("Stufoase");

    return results;
}

// ---------- Barbă și mustață ----------
function classifyBeardAndMustache(landmarks, canvas, ctx) {
    // Referință piele (obraz)
    const skinRef = landmarks[LM.RIGHT_CHEEK_SKIN];
    const sx = Math.round(clamp(skinRef.x, 0, 1) * canvas.width);
    const sy = Math.round(clamp(skinRef.y, 0, 1) * canvas.height);
    const skinPixel = ctx.getImageData(clamp(sx, 0, canvas.width - 1), clamp(sy, 0, canvas.height - 1), 1, 1).data;
    const skinLum = (skinPixel[0] + skinPixel[1] + skinPixel[2]) / 3;

    // Regiunea bărbii
    const chin = landmarks[LM.CHIN];
    const rightJaw = landmarks[LM.RIGHT_JAW];
    const leftJaw = landmarks[LM.LEFT_JAW];
    const beardSamples = [];
    for (let t = 0; t <= 1; t += 0.2) {
        const x = rightJaw.x + (leftJaw.x - rightJaw.x) * t;
        const y = rightJaw.y + (chin.y - rightJaw.y) * t * 1.5;
        const cx = Math.round(clamp(x, 0.05, 0.95) * canvas.width);
        const cy = Math.round(clamp(y, 0.1, 0.95) * canvas.height);
        for (let dx = -2; dx <= 2; dx++) {
            for (let dy = -2; dy <= 2; dy++) {
                const px = clamp(cx + dx, 0, canvas.width - 1);
                const py = clamp(cy + dy, 0, canvas.height - 1);
                const pixel = ctx.getImageData(px, py, 1, 1).data;
                beardSamples.push((pixel[0] + pixel[1] + pixel[2]) / 3);
            }
        }
    }
    const beardAvgLum = beardSamples.reduce((a, b) => a + b, 0) / beardSamples.length;
    const beardDarkRatio = beardSamples.filter((v) => v < skinLum - 40).length / beardSamples.length;

    let barba = "Fără barbă";
    if (beardDarkRatio > 0.55) barba = "Barbă completă";
    else if (beardDarkRatio > 0.35) barba = "Barbă medie";
    else if (beardDarkRatio > 0.18) barba = "Barbă scurtă";

    // Mustață
    const mouthTop = landmarks[LM.MOUTH_TOP];
    const mustacheSamples = [];
    for (let t = 0; t <= 1; t += 0.15) {
        const x = landmarks[LM.MOUTH_RIGHT].x + (landmarks[LM.MOUTH_LEFT].x - landmarks[LM.MOUTH_RIGHT].x) * t;
        const y = mouthTop.y - 0.015;
        const cx = Math.round(clamp(x, 0.05, 0.95) * canvas.width);
        const cy = Math.round(clamp(y, 0.05, 0.95) * canvas.height);
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -2; dy <= 1; dy++) {
                const px = clamp(cx + dx, 0, canvas.width - 1);
                const py = clamp(cy + dy, 0, canvas.height - 1);
                const pixel = ctx.getImageData(px, py, 1, 1).data;
                mustacheSamples.push((pixel[0] + pixel[1] + pixel[2]) / 3);
            }
        }
    }
    const mustacheAvgLum = mustacheSamples.reduce((a, b) => a + b, 0) / mustacheSamples.length;
    const mustacheDarkRatio = mustacheSamples.filter((v) => v < skinLum - 30).length / mustacheSamples.length;

    let mustata = "Fără mustață";
    if (mustacheDarkRatio > 0.5) mustata = "Groasă";
    else if (mustacheDarkRatio > 0.3) mustata = "Subțire";

    return { barba, mustata };
}

// ---------- Urechi ----------
function classifyEars(landmarks, faceWidth, faceHeight, profileLandmarks = null) {
    // Folosim landmark-urile din profil dacă există, altfel frontal
    const sourceLandmarks = profileLandmarks || landmarks;

    const earTopRight = sourceLandmarks[LM.RIGHT_EAR_TOP];
    const earBottomRight = sourceLandmarks[LM.RIGHT_EAR_BOTTOM];
    const earLobeRight = sourceLandmarks[LM.RIGHT_EAR_LOBE];
    const earTopLeft = sourceLandmarks[LM.LEFT_EAR_TOP];
    const earBottomLeft = sourceLandmarks[LM.LEFT_EAR_BOTTOM];
    const earLobeLeft = sourceLandmarks[LM.LEFT_EAR_LOBE];

    if (!earTopRight || !earBottomRight || !earTopLeft || !earBottomLeft) {
        return { forma: "Nedeterminată", marime: "Nedeterminată", lob: "Nedeterminat" };
    }

    // Mărime
    const rightEarHeight = distance(earTopRight, earBottomRight);
    const leftEarHeight = distance(earTopLeft, earBottomLeft);
    const avgEarHeight = (rightEarHeight + leftEarHeight) / 2;
    const faceHeightTotal = faceHeight || distance(sourceLandmarks[LM.HAIRLINE_CENTER], sourceLandmarks[LM.CHIN]);
    const earHeightRatio = avgEarHeight / faceHeightTotal;

    let marime;
    if (earHeightRatio < 0.22) marime = "Mici";
    else if (earHeightRatio > 0.32) marime = "Mari";
    else marime = "Medii";

    // Formă: raport înălțime / lățime (aproximez lățimea)
    const rightTragus = sourceLandmarks[LM.FACE_RIGHT_TEMPLE];
    const leftTragus = sourceLandmarks[LM.FACE_LEFT_TEMPLE];
    const rightEarWidth = distance(rightTragus, earTopRight) * 1.5;
    const leftEarWidth = distance(leftTragus, earTopLeft) * 1.5;
    const avgEarWidth = (rightEarWidth + leftEarWidth) / 2;
    const ratio = avgEarHeight / avgEarWidth;

    let forma;
    if (ratio > 1.8) forma = "Dreptunghiulară";
    else if (ratio > 1.4) forma = "Ovală";
    else if (ratio > 1.0) forma = "Rotundă";
    else if (ratio > 0.7) forma = "Triunghiulară";
    else forma = "Neregulată";

    // Lob
    const rightLobeAngle = Math.atan2(earBottomRight.y - earLobeRight.y, earBottomRight.x - earLobeRight.x);
    const leftLobeAngle = Math.atan2(earBottomLeft.y - earLobeLeft.y, earBottomLeft.x - earLobeLeft.x);
    const avgLobeAngle = (rightLobeAngle + leftLobeAngle) / 2;

    let lob;
    if (Math.abs(avgLobeAngle) < 0.3) lob = "Atașat";
    else if (Math.abs(avgLobeAngle) > 0.8) lob = "Liber";
    else lob = "Nedeterminat";

    return { forma, marime, lob };
}

// ============================================================
// FLUX PRINCIPAL DE ANALIZĂ
// ============================================================
async function runAnalysis() {
    const statusEl = document.getElementById("status");
    statusEl.className = "status info";
    statusEl.textContent = "⏳ Se procesează imaginile...";

    const btn = document.getElementById("btn-analyze");
    btn.disabled = true;
    document.getElementById("analyze-spinner").style.display = "inline-block";
    document.getElementById("analyze-text").textContent = "Se analizează...";

    try {
        // Procesăm imaginea frontală
        if (!frontalFile) throw new Error("Încarcă poza din față.");
        const frontalProc = await processImage(frontalFile);
        const frontalLandmarks = await extractLandmarks(frontalProc);
        if (!frontalLandmarks) {
            statusEl.className = "status error";
            statusEl.textContent =
                "❌ Nu s-au putut detecta landmark-urile faciale în poza din față. Verifică încadrarea și claritatea.";
            return;
        }
        frontalImageData = { ...frontalProc, landmarks: frontalLandmarks };

        // Procesăm imaginea de profil (opțional)
        let profilLandmarks = null;
        if (profilFile) {
            try {
                const profilProc = await processImage(profilFile);
                profilLandmarks = await extractLandmarks(profilProc);
                profilImageData = { ...profilProc, landmarks: profilLandmarks };
                if (!profilLandmarks) {
                    console.warn("⚠️ Nu s-au detectat landmark-uri în poza de profil. Folosim doar analiza frontală.");
                }
            } catch (profilErr) {
                console.warn("Eroare la procesarea pozei de profil:", profilErr);
            }
        }

        // Calculăm dimensiunile de referință
        const faceWidth = distance(
            frontalLandmarks[LM.FACE_RIGHT_TEMPLE],
            frontalLandmarks[LM.FACE_LEFT_TEMPLE]
        );
        const faceHeight = distance(frontalLandmarks[LM.HAIRLINE_CENTER], frontalLandmarks[LM.CHIN]);

        // Construim obiectul de rezultate
        const barbaMustata = classifyBeardAndMustache(frontalLandmarks, frontalProc.canvas, frontalProc.ctx);
        const urechi = classifyEars(frontalLandmarks, faceWidth, faceHeight, profilLandmarks);

        const results = {
            frunte: classifyForehead(frontalLandmarks, faceWidth),
            nas: classifyNose(frontalLandmarks, profilLandmarks),
            ochi: classifyEyes(frontalLandmarks, frontalProc.canvas, frontalProc.ctx, faceWidth),
            gura: classifyMouth(frontalLandmarks, faceWidth),
            barbie: classifyChin(frontalLandmarks, faceWidth, frontalProc.canvas, frontalProc.ctx),
            tipFata: classifyFaceType(frontalLandmarks),
            par: classifyHair(frontalLandmarks, frontalProc.canvas, frontalProc.ctx),
            sprancene: classifyEyebrows(frontalLandmarks, frontalProc.canvas, frontalProc.ctx),
            barba: barbaMustata.barba,
            mustata: barbaMustata.mustata,
            urechi: urechi,
            semneParticulare: "",
        };

        currentResults = results;
        renderResults(results);
        document.getElementById("results-section").classList.add("visible");
        statusEl.className = "status success";
        statusEl.textContent = "✅ Analiza completă! Verifică și corectează rezultatele, apoi salvează fișa.";

        // Cleanup imaginilor din memorie
        clearCanvasData(frontalProc);
        if (profilImageData) clearCanvasData(profilImageData);

        console.log("🔄 Imaginile au fost eliminate din canvas/memorie. Doar rezultatele JSON sunt păstrate.");
    } catch (err) {
        console.error("Eroare în analiză:", err);
        statusEl.className = "status error";
        statusEl.textContent = "❌ Eroare: " + err.message;
    } finally {
        btn.disabled = false;
        document.getElementById("analyze-spinner").style.display = "none";
        document.getElementById("analyze-text").textContent = "🔬 Analizează fețele";
        checkAnalyzeButton();
    }
}

// ============================================================
// RENDERIZARE REZULTATE (dropdown-uri și checkbox-uri)
// ============================================================
function renderResults(results) {
    const grid = document.getElementById("results-grid");
    grid.innerHTML = "";

    // Card 1: Frunte
    grid.appendChild(
        createCard("👤", "Fruntea", [
            makeSelect("frunte-tip", "Tipul frunții", OPTIONS.frunte, results.frunte.tip || "Mijlocie"),
        ])
    );

    // Card 2: Nas
    grid.appendChild(
        createCard("👃", "Nasul", [
            makeSelect("nas-tip", "Tipul nasului", OPTIONS.nas, results.nas || "Drept"),
        ])
    );

    // Card 3: Ochi
    grid.appendChild(
        createCard("👁️", "Ochii", [
            makeSelect("ochi-culoare", "Culoarea ochilor", OPTIONS.ochiCuloare, results.ochi.culoare || "Căprui"),
            makeSelect("ochi-marime", "Mărimea ochilor", OPTIONS.ochiMarime, results.ochi.marime || "Mijlocii"),
        ])
    );

    // Card 4: Gură
    grid.appendChild(
        createCard("👄", "Gura", [
            makeSelect("gura-colturi", "Colțurile gurii", OPTIONS.guraColturi, results.gura.colturi || "Liniară"),
            makeSelect("gura-marime", "Mărimea gurii", OPTIONS.guraMarime, results.gura.marime || "Mijlocie"),
        ])
    );

    // Card 5: Bărbie
    grid.appendChild(
        createCard("🫦", "Bărbia", [
            makeSelect("barbie-tip", "Tipul bărbiei", OPTIONS.barbie, results.barbie.tip || "Normală"),
        ])
    );

    // Card 6: Tipul feței
    grid.appendChild(
        createCard("📐", "Tipul feței", [
            makeSelect("tip-fata", "Forma feței", OPTIONS.tipFata, results.tipFata.tip || "Ovală"),
        ])
    );

    // Card 7: Păr
    grid.appendChild(
        createCard("💇", "Părul", [
            makeSelect("par-culoare", "Culoarea părului", OPTIONS.parCuloare, results.par.culoare || "Șaten"),
            makeSelect("par-textura", "Textura părului", OPTIONS.parTextura, results.par.textura || "Drept"),
            makeSelect("par-calvitie", "Calviția", OPTIONS.calvitie, results.par.calvitie || "Fără calviție"),
        ])
    );

    // Card 8: Sprâncene
    grid.appendChild(
        createCard("🖤", "Sprâncenele", [
            makeCheckboxGroup("sprancene-opts", OPTIONS.sprancene, results.sprancene || ["Drepte"]),
        ])
    );

    // Card 9: Barbă
    grid.appendChild(
        createCard("🧔", "Barba", [
            makeSelect("barba-tip", "Tipul bărbii", OPTIONS.barba, results.barba || "Fără barbă"),
        ])
    );

    // Card 10: Mustață
    grid.appendChild(
        createCard("👨", "Mustața", [
            makeSelect("mustata-tip", "Tipul mustății", OPTIONS.mustata, results.mustata || "Fără mustață"),
        ])
    );

    // Card 11: Urechi
    grid.appendChild(
        createCard("👂", "Urechile", [
            makeSelect("urechi-forma", "Forma urechii", OPTIONS.urechiForma, results.urechi.forma || "Nedeterminată"),
            makeSelect("urechi-marime", "Mărimea urechii", OPTIONS.urechiMarime, results.urechi.marime || "Nedeterminată"),
            makeSelect("urechi-lob", "Lobul urechii", OPTIONS.urechiLob, results.urechi.lob || "Nedeterminat"),
        ])
    );

    // Card 12: Semne particulare
    grid.appendChild(
        createCard("⭐", "Semne particulare", [
            makeTextInput("semne-text", "Tatuaje, cicatrici etc.", results.semneParticulare || ""),
        ])
    );

    // Card informativ despre fiabilitate
    const infoCard = document.createElement("div");
    infoCard.className = "result-card";
    infoCard.style.borderColor = "rgba(255,255,255,0.15)";
    infoCard.innerHTML = `
        <div class="card-title"><span class="emoji">ℹ️</span> Fiabilitate</div>
        <p style="font-size:0.8rem;color:var(--text-secondary);">
            Categoriile geometrice (tip față, gură, frunte, sprâncene) au fiabilitate ridicată.
            Culoarea ochilor/părului este aproximativă (sampling RGB). Tipul nasului necesită
            poza de profil pentru o precizie mai bună. Barba/mustața sunt orientative.
            Verifică toate rezultatele înainte de salvare.
        </p>
    `;
    grid.appendChild(infoCard);
}

function createCard(emoji, title, fields) {
    const card = document.createElement("div");
    card.className = "result-card";
    const titleEl = document.createElement("div");
    titleEl.className = "card-title";
    titleEl.innerHTML = `<span class="emoji">${emoji}</span> ${title}`;
    card.appendChild(titleEl);
    fields.forEach((f) => card.appendChild(f));
    return card;
}

function makeSelect(id, labelText, options, selectedValue) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("label");
    label.textContent = labelText;
    const select = document.createElement("select");
    select.id = id;
    select.setAttribute("data-category", id);
    options.forEach((opt) => {
        const option = document.createElement("option");
        option.value = opt;
        option.textContent = opt;
        if (opt === selectedValue) option.selected = true;
        select.appendChild(option);
    });
    wrapper.appendChild(label);
    wrapper.appendChild(select);
    return wrapper;
}

function makeCheckboxGroup(id, options, selectedValues) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("label");
    label.textContent = "Caracteristici (selectează toate care se aplică)";
    const group = document.createElement("div");
    group.className = "checkbox-group";
    group.id = id;
    options.forEach((opt) => {
        const cbLabel = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = opt;
        cb.checked = selectedValues.includes(opt);
        cbLabel.appendChild(cb);
        cbLabel.appendChild(document.createTextNode(opt));
        group.appendChild(cbLabel);
    });
    wrapper.appendChild(label);
    wrapper.appendChild(group);
    return wrapper;
}

function makeTextInput(id, labelText, value) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";
    const label = document.createElement("label");
    label.textContent = labelText;
    const input = document.createElement("input");
    input.type = "text";
    input.id = id;
    input.value = value;
    input.placeholder = "Introdu manual observații...";
    wrapper.appendChild(label);
    wrapper.appendChild(input);
    return wrapper;
}

// ============================================================
// COLECTARE REZULTATE DIN UI
// ============================================================
function collectResultsFromUI() {
    const getVal = (id) => document.getElementById(id)?.value;
    const getCheckboxVals = (id) => {
        const group = document.getElementById(id);
        if (!group) return [];
        return Array.from(group.querySelectorAll('input[type="checkbox"]:checked')).map((cb) => cb.value);
    };

    return {
        frunte: { tip: getVal("frunte-tip") },
        nas: getVal("nas-tip"),
        ochi: {
            culoare: getVal("ochi-culoare"),
            marime: getVal("ochi-marime"),
            lipsa: false,
        },
        gura: {
            colturi: getVal("gura-colturi"),
            marime: getVal("gura-marime"),
        },
        barbie: { tip: getVal("barbie-tip") },
        tipFata: { tip: getVal("tip-fata") },
        par: {
            culoare: getVal("par-culoare"),
            textura: getVal("par-textura"),
            calvitie: getVal("par-calvitie"),
        },
        sprancene: getCheckboxVals("sprancene-opts"),
        barba: getVal("barba-tip"),
        mustata: getVal("mustata-tip"),
        urechi: {
            forma: getVal("urechi-forma"),
            marime: getVal("urechi-marime"),
            lob: getVal("urechi-lob"),
        },
        semneParticulare: getVal("semne-text") || "",
        dataAnaliza: new Date().toISOString(),
    };
}

// ============================================================
// SALVARE LOCALSTORAGE + EXPORT
// ============================================================
function saveResults() {
    const data = collectResultsFromUI();
    const key = "semnalmente:" + Date.now();
    try {
        localStorage.setItem(key, JSON.stringify(data));
        alert("✅ Fișa a fost salvată în localStorage sub cheia: " + key);
        renderSavedList();
    } catch (err) {
        alert("❌ Eroare la salvare: " + err.message);
    }
}

function exportResults() {
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
    keys.forEach((key) => {
        const raw = localStorage.getItem(key);
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch {
            parsed = { dataAnaliza: "necunoscută" };
        }
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
        loadBtn.textContent = "📂 Încarcă";
        loadBtn.addEventListener("click", () => {
            loadSavedData(parsed);
        });
        const delBtn = document.createElement("button");
        delBtn.textContent = "🗑️ Șterge";
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
    currentResults = data;
    renderResults(data);
    document.getElementById("results-section").classList.add("visible");
    document.getElementById("status").className = "status info";
    document.getElementById("status").textContent = "📂 Fișă încărcată. Verifică și corectează dacă e necesar.";
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
        // Reset previews
        ["frontal", "profil"].forEach((prefix) => {
            const preview = document.getElementById(`preview-${prefix}`);
            preview.src = "";
            preview.classList.remove("visible");
            document.getElementById(`remove-${prefix}`).classList.remove("visible");
            document.getElementById(`file-${prefix}`).value = "";
        });
        objectUrls.forEach((url) => URL.revokeObjectURL(url));
        objectUrls = [];
        checkAnalyzeButton();
        renderSavedList();
    }
}

// ============================================================
// INIȚIALIZARE APLICAȚIE
// ============================================================
async function initApp() {
    setupUploads();
    checkAnalyzeButton();
    renderSavedList();

    document.getElementById("btn-analyze").addEventListener("click", runAnalysis);
    document.getElementById("btn-save").addEventListener("click", saveResults);
    document.getElementById("btn-export").addEventListener("click", exportResults);
    document.getElementById("btn-reset").addEventListener("click", resetAll);

    // Inițializăm MediaPipe în fundal
    try {
        await initFaceLandmarker();
        console.log("🚀 Aplicație pregătită. Poți încărca imagini.");
    } catch (err) {
        console.error("Eroare la inițializarea MediaPipe:", err);
        document.getElementById("status").className = "status error";
        document.getElementById("status").textContent =
            "⚠️ Eroare la încărcarea modelului MediaPipe. Verifică conexiunea la internet (CDN-urile trebuie să fie accesibile).";
    }
}

// Pornire aplicație
initApp();