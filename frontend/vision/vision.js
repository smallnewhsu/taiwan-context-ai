const fileInput = document.querySelector("#imageFile");
const analyzeButton = document.querySelector("#analyzeButton");
const imagePreview = document.querySelector("#imagePreview");
const fileInfo = document.querySelector("#fileInfo");
const progress = document.querySelector("#progress");
const errorMessage = document.querySelector("#errorMessage");
const resultPanel = document.querySelector("#resultPanel");
const dropZone = document.querySelector("#dropZone");
let previewUrl = null;
let latestVisionAnalysis = null;

const show = (element) => element.classList.remove("hidden");
const hide = (element) => element.classList.add("hidden");
const percent = (value) => `${Math.round(Number(value) * 100)}%`;

function selectedFile() {
  return fileInput.files && fileInput.files[0];
}

function updateFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    errorMessage.textContent = "請選擇 JPG、PNG 或 WEBP 圖片。";
    show(errorMessage);
    analyzeButton.disabled = true;
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    errorMessage.textContent = "圖片超過 15 MB 限制。";
    show(errorMessage);
    analyzeButton.disabled = true;
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileInfo.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  show(fileInfo);
  show(imagePreview);
  hide(errorMessage);
  hide(resultPanel);
  analyzeButton.disabled = false;
}

fileInput.addEventListener("change", () => updateFile(selectedFile()));

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  updateFile(file);
});

analyzeButton.addEventListener("click", async () => {
  const file = selectedFile();
  if (!file) return;
  analyzeButton.disabled = true;
  show(progress);
  hide(errorMessage);
  hide(resultPanel);

  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("/vision/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "圖片理解失敗");
    renderResult(payload);
  } catch (error) {
    errorMessage.textContent = error.message;
    show(errorMessage);
  } finally {
    analyzeButton.disabled = false;
    hide(progress);
  }
});

function confidenceBadge(value, threshold) {
  const badge = document.createElement("span");
  badge.className = `confidence${Number(value) < threshold ? " low" : ""}`;
  badge.textContent = percent(value);
  return badge;
}

function renderConfidenceList(target, items, threshold) {
  target.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "empty-item";
    empty.textContent = "沒有足夠清楚的內容";
    target.append(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("li");
    row.append(document.createTextNode(item.text));
    row.append(confidenceBadge(item.confidence, threshold));
    target.append(row);
  });
}

function renderInferences(items) {
  const target = document.querySelector("#inferenceList");
  target.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-item";
    empty.textContent = "沒有足夠證據提出情境推測";
    target.append(empty);
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "inference-item";
    const title = document.createElement("strong");
    title.append(document.createTextNode(item.description));
    title.append(confidenceBadge(item.confidence, 0.75));
    const basis = document.createElement("p");
    basis.textContent = `圖片依據：${item.basis || "模型未提供明確依據"}`;
    card.append(title, basis);
    target.append(card);
  });
}

function renderStringList(target, items) {
  target.replaceChildren();
  const values = items.length ? items : ["未列出"];
  values.forEach((text) => {
    const row = document.createElement("li");
    row.textContent = text;
    target.append(row);
  });
}

const warningLabels = {
  low_confidence_ocr_requires_confirmation: "部分圖片文字可信度不足，請對照原圖確認。",
  context_is_inference_not_fact: "生活情境屬於推測，不能當成圖片中的既定事實。",
  low_confidence_context_requires_confirmation: "部分情境推測缺乏足夠證據。",
  low_confidence_observation: "部分物件辨識可信度偏低。",
};

function renderWarnings(warnings) {
  const box = document.querySelector("#warningBox");
  const list = document.querySelector("#warningList");
  list.replaceChildren();
  if (!warnings.length) {
    hide(box);
    return;
  }
  warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warningLabels[warning] || warning;
    list.append(item);
  });
  show(box);
}

function renderResult(data) {
  const analysis = data.analysis;
  latestVisionAnalysis = analysis;
  renderConfidenceList(document.querySelector("#observationList"), analysis.observations, 0.65);
  renderConfidenceList(document.querySelector("#textList"), analysis.visible_text, 0.85);
  renderInferences(analysis.context_inferences);
  renderStringList(document.querySelector("#uncertaintyList"), analysis.uncertainties);
  renderWarnings(data.warnings || []);

  const badge = document.querySelector("#statusBadge");
  if (data.needs_confirmation) {
    document.querySelector("#statusTitle").textContent = "完成，部分內容待確認";
    badge.textContent = "請確認";
    badge.classList.add("warning");
  } else {
    document.querySelector("#statusTitle").textContent = "已完成安全分析";
    badge.textContent = "低風險";
    badge.classList.remove("warning");
  }

  document.querySelector("#modelName").textContent = data.model;
  document.querySelector("#processingTime").textContent = `${Number(data.processing_seconds).toFixed(3)} 秒`;
  document.querySelector("#originalSize").textContent = `${data.original_size.width} × ${data.original_size.height}`;
  document.querySelector("#processedSize").textContent = `${data.processed_size.width} × ${data.processed_size.height}`;
  document.querySelector("#localProcessing").textContent = data.privacy.processed_locally ? "是" : "否";
  document.querySelector("#imagePersisted").textContent = data.privacy.image_persisted ? "是" : "否";

  show(resultPanel);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

const followupAudio = document.querySelector("#followupAudio");
const askFollowupButton = document.querySelector("#askFollowupButton");
const followupText = document.querySelector("#followupText");
const recordButton = document.querySelector("#recordButton");
const stopButton = document.querySelector("#stopButton");
const resetRecordingButton = document.querySelector("#resetRecordingButton");
const recordingPreview = document.querySelector("#recordingPreview");
const recordingPreviewBox = document.querySelector("#recordingPreviewBox");
const recordingTime = document.querySelector("#recordingTime");
let questionMode = "record";
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedAudioBlob = null;
let recordingUrl = null;
let recordingTimer = null;
let recordingSeconds = 0;

document.querySelectorAll(".question-tab").forEach((button) => {
  button.addEventListener("click", () => {
    questionMode = button.dataset.mode;
    document.querySelectorAll(".question-tab").forEach((tab) => {
      const active = tab === button;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    document.querySelector("#recordMode").classList.toggle("hidden", questionMode !== "record");
    document.querySelector("#uploadMode").classList.toggle("hidden", questionMode !== "upload");
    document.querySelector("#textMode").classList.toggle("hidden", questionMode !== "text");
    hide(document.querySelector("#followupError"));
  });
});

function updateRecordingTime() {
  const minutes = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
  const seconds = String(recordingSeconds % 60).padStart(2, "0");
  recordingTime.textContent = `${minutes}:${seconds}`;
}

function releaseMicrophone() {
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
}

function resetRecording() {
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  releaseMicrophone();
  clearInterval(recordingTimer);
  recordedChunks = [];
  recordedAudioBlob = null;
  recordingSeconds = 0;
  updateRecordingTime();
  if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  recordingUrl = null;
  recordingPreview.removeAttribute("src");
  hide(recordingPreviewBox);
  recordButton.disabled = false;
  stopButton.disabled = true;
  recordButton.textContent = "開始錄音";
}

recordButton.addEventListener("click", async () => {
  const followupError = document.querySelector("#followupError");
  hide(followupError);
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    followupError.textContent = "瀏覽器無法使用麥克風。請以 http://127.0.0.1:8000/app/vision/ 或 HTTPS 開啟系統。";
    show(followupError);
    return;
  }
  try {
    resetRecording();
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
    mediaRecorder = new MediaRecorder(mediaStream, {mimeType});
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) recordedChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      recordedAudioBlob = new Blob(recordedChunks, {type: mediaRecorder.mimeType || "audio/webm"});
      recordingUrl = URL.createObjectURL(recordedAudioBlob);
      recordingPreview.src = recordingUrl;
      show(recordingPreviewBox);
      releaseMicrophone();
    });
    mediaRecorder.start();
    recordingTimer = setInterval(() => {
      recordingSeconds += 1;
      updateRecordingTime();
      if (recordingSeconds >= 60) stopButton.click();
    }, 1000);
    recordButton.disabled = true;
    stopButton.disabled = false;
    recordButton.textContent = "錄音中…";
  } catch (error) {
    releaseMicrophone();
    followupError.textContent = error.name === "NotAllowedError" ? "麥克風權限遭拒，請在瀏覽器網址列允許麥克風後重試。" : `無法開始錄音：${error.message}`;
    show(followupError);
  }
});

stopButton.addEventListener("click", () => {
  if (mediaRecorder?.state !== "recording") return;
  mediaRecorder.stop();
  clearInterval(recordingTimer);
  recordButton.disabled = false;
  stopButton.disabled = true;
  recordButton.textContent = "開始錄音";
});

resetRecordingButton.addEventListener("click", resetRecording);
followupAudio.addEventListener("change", () => {
  document.querySelector("#audioFileName").textContent = followupAudio.files[0] ? `已選擇：${followupAudio.files[0].name}` : "尚未選擇音檔";
});

async function parseResponse(response) {
  let payload;
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || "多模態問答失敗");
  return payload;
}

async function submitAudioQuestion(audio) {
  const form = new FormData();
  form.append("audio", audio);
  form.append("image_context", JSON.stringify(latestVisionAnalysis));
  form.append("relationship", document.querySelector("#followupRelationship").value);
  return parseResponse(await fetch("/vision/follow-up", {method: "POST", body: form}));
}

async function submitTextQuestion() {
  const question = followupText.value.trim();
  if (!question) throw new Error("請先輸入問題。");
  return parseResponse(await fetch("/vision/follow-up/text", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question, image_context: latestVisionAnalysis, relationship: document.querySelector("#followupRelationship").value}),
  }));
}

function renderFollowup(payload) {
  document.querySelector("#followupQuestion").textContent = payload.question;
  document.querySelector("#followupAnswer").textContent = payload.answer.answer;
  const basis = document.querySelector("#followupBasis");
  const values = payload.answer.basis?.length ? payload.answer.basis : ["依據目前影像分析與使用者問題回答"];
  basis.replaceChildren(...values.map((text) => { const item = document.createElement("li"); item.textContent = text; return item; }));
  const warning = document.querySelector("#followupWarning");
  const flags = [];
  if (payload.question_needs_confirmation) flags.push("語音辨識候選不一致，請確認問題文字");
  if (payload.answer.needs_confirmation) flags.push(`仍無法確認：${(payload.answer.uncertainties || []).join("、")}`);
  warning.textContent = flags.join("；");
  warning.classList.toggle("hidden", !flags.length);
  show(document.querySelector("#followupResult"));
}

askFollowupButton.addEventListener("click", async () => {
  const followupError = document.querySelector("#followupError");
  if (!latestVisionAnalysis) {
    followupError.textContent = "請先完成圖片分析。";
    show(followupError);
    return;
  }
  askFollowupButton.disabled = true;
  show(document.querySelector("#followupProgress"));
  hide(followupError);
  hide(document.querySelector("#followupResult"));
  try {
    let payload;
    if (questionMode === "record") {
      if (!recordedAudioBlob) throw new Error("請先錄製問題。");
      payload = await submitAudioQuestion(new File([recordedAudioBlob], "recorded-question.webm", {type: recordedAudioBlob.type || "audio/webm"}));
    } else if (questionMode === "upload") {
      const audio = followupAudio.files[0];
      if (!audio) throw new Error("請先選擇問題音檔。");
      payload = await submitAudioQuestion(audio);
    } else {
      payload = await submitTextQuestion();
    }
    renderFollowup(payload);
  } catch (error) {
    followupError.textContent = error.message;
    show(followupError);
  } finally {
    askFollowupButton.disabled = false;
    hide(document.querySelector("#followupProgress"));
  }
});
