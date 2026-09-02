const fileInput = document.querySelector("#imageFile");
const analyzeButton = document.querySelector("#analyzeButton");
const imagePreview = document.querySelector("#imagePreview");
const fileInfo = document.querySelector("#fileInfo");
const progress = document.querySelector("#progress");
const errorMessage = document.querySelector("#errorMessage");
const resultPanel = document.querySelector("#resultPanel");
const dropZone = document.querySelector("#dropZone");
let previewUrl = null;

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
