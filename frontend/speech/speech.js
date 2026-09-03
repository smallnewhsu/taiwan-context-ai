const fileInput = document.querySelector("#audioFile");
const analyzeButton = document.querySelector("#analyzeButton");
const audioPreview = document.querySelector("#audioPreview");
const fileInfo = document.querySelector("#fileInfo");
const progress = document.querySelector("#progress");
const errorMessage = document.querySelector("#errorMessage");
const resultPanel = document.querySelector("#resultPanel");
let latestResult = null;

const show = (element) => element.classList.remove("hidden");
const hide = (element) => element.classList.add("hidden");
const round = (value, digits = 3) => Number(value).toFixed(digits);

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileInfo.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  audioPreview.src = URL.createObjectURL(file);
  show(fileInfo);
  show(audioPreview);
  analyzeButton.disabled = false;
  hide(errorMessage);
  hide(resultPanel);
});

analyzeButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  analyzeButton.disabled = true;
  show(progress);
  hide(errorMessage);
  hide(resultPanel);

  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("/speech/interpret", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "語音處理失敗");
    latestResult = payload;
    renderResult(payload);
  } catch (error) {
    errorMessage.textContent = error.message;
    show(errorMessage);
  } finally {
    analyzeButton.disabled = false;
    hide(progress);
  }
});

function renderResult(data) {
  const decision = data.decision;
  document.querySelector("#audioDuration").textContent = `${round(data.audio_duration_seconds)} 秒`;
  document.querySelector("#processingTime").textContent = `${round(data.dual_pass_seconds)} 秒`;
  document.querySelector("#baselineConfidence").textContent = round(data.baseline.confidence, 4);
  document.querySelector("#promptConfidence").textContent = round(data.prompt.confidence, 4);
  document.querySelector("#decisionReasons").textContent = decision.reasons.join("、");
  document.querySelector("#baselineText").textContent = data.baseline.text;
  document.querySelector("#promptText").textContent = data.prompt.text;
  document.querySelector("#selectedText").textContent = decision.selected_text;
  document.querySelector("#manualText").value = "";
  document.querySelector("#datasetConsent").checked = false;
  document.querySelectorAll('input[name="candidate"]').forEach((input) => {
    input.checked = input.value === decision.selected_source;
  });
  hide(document.querySelector("#confirmedMessage"));

  const badge = document.querySelector("#statusBadge");
  if (decision.needs_confirmation) {
    document.querySelector("#statusTitle").textContent = "需要您的確認";
    badge.textContent = "低信心";
    badge.classList.add("warning");
    hide(document.querySelector("#safeResult"));
    show(document.querySelector("#confirmationBox"));
  } else {
    document.querySelector("#statusTitle").textContent = "已完成理解";
    badge.textContent = "可自動使用";
    badge.classList.remove("warning");
    show(document.querySelector("#safeResult"));
    hide(document.querySelector("#confirmationBox"));
  }
  show(resultPanel);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.querySelector("#confirmButton").addEventListener("click", async () => {
  const manualText = document.querySelector("#manualText").value.trim();
  const selected = document.querySelector('input[name="candidate"]:checked');
  let confirmedText = manualText;
  if (!confirmedText && selected && latestResult) {
    confirmedText = latestResult[selected.value].text;
  }
  if (!confirmedText) {
    alert("請選擇候選內容或輸入修正文字。");
    return;
  }
  const button = document.querySelector("#confirmButton");
  button.disabled = true;
  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_file: latestResult.filename,
        baseline_text: latestResult.baseline.text,
        prompt_text: latestResult.prompt.text,
        selected_source: manualText ? "manual" : selected.value,
        corrected_text: confirmedText,
        reasons: latestResult.decision.reasons,
        consent_to_dataset: document.querySelector("#datasetConsent").checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法儲存確認內容");
    const message = document.querySelector("#confirmedMessage");
    message.textContent = `已確認並送交審核：${confirmedText}`;
    show(message);
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
});
