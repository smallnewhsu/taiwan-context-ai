const $ = (selector) => document.querySelector(selector);
const fileInput = $("#audioFile");
const analyzeButton = $("#analyzeButton");
const audioPreview = $("#audioPreview");
const fileInfo = $("#fileInfo");
const progress = $("#progress");
const errorMessage = $("#errorMessage");
const resultPanel = $("#resultPanel");
const show = (element) => element.classList.remove("hidden");
const hide = (element) => element.classList.add("hidden");
const round = (value, digits = 3) => Number(value).toFixed(digits);
let latestResult = null;
let selectedAudio = null;
let selectedAudioUrl = null;
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordingTimer = null;
let recordingSeconds = 0;

const recordButton = $("#recordButton");
const stopRecordButton = $("#stopRecordButton");
const discardRecordButton = $("#discardRecordButton");
const recordTime = $("#recordTime");

function setAudioSource(file, label) {
  selectedAudio = file;
  if (selectedAudioUrl) URL.revokeObjectURL(selectedAudioUrl);
  selectedAudioUrl = URL.createObjectURL(file);
  audioPreview.src = selectedAudioUrl;
  audioPreview.load();
  fileInfo.textContent = label;
  show(audioPreview);
  show(discardRecordButton);
  analyzeButton.disabled = false;
  hide(errorMessage);
  hide(resultPanel);
}

function updateRecordTime() {
  const minutes = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
  const seconds = String(recordingSeconds % 60).padStart(2, "0");
  recordTime.textContent = `${minutes}:${seconds}`;
}

function releaseMicrophone() {
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
}

function clearSelectedAudio() {
  selectedAudio = null;
  fileInput.value = "";
  if (selectedAudioUrl) URL.revokeObjectURL(selectedAudioUrl);
  selectedAudioUrl = null;
  audioPreview.removeAttribute("src");
  audioPreview.load();
  hide(audioPreview);
  hide(discardRecordButton);
  analyzeButton.disabled = true;
  fileInfo.textContent = "點擊麥克風直接錄音，或選擇既有音檔";
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  setAudioSource(file, `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`);
});

recordButton.addEventListener("click", async () => {
  hide(errorMessage);
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    errorMessage.textContent = "無法使用麥克風。請以 http://127.0.0.1:8000/app/speech/ 或 HTTPS 開啟系統。";
    show(errorMessage);
    return;
  }
  try {
    clearSelectedAudio();
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      const file = new File([blob], "speech-recording.webm", { type: blob.type });
      setAudioSource(file, `錄音完成 · ${recordingSeconds} 秒`);
      releaseMicrophone();
    });
    recordingSeconds = 0;
    updateRecordTime();
    mediaRecorder.start();
    recordingTimer = setInterval(() => {
      recordingSeconds += 1;
      updateRecordTime();
      if (recordingSeconds >= 60) stopRecordButton.click();
    }, 1000);
    recordButton.disabled = true;
    stopRecordButton.disabled = false;
    recordButton.classList.add("recording");
    recordButton.setAttribute("aria-label", "錄音中");
    fileInfo.textContent = "錄音中，完成後請按停止錄音";
  } catch (error) {
    releaseMicrophone();
    errorMessage.textContent = error.name === "NotAllowedError" ? "麥克風權限遭拒，請在瀏覽器網址列允許麥克風後重試。" : `無法開始錄音：${error.message}`;
    show(errorMessage);
  }
});

stopRecordButton.addEventListener("click", () => {
  if (mediaRecorder?.state !== "recording") return;
  mediaRecorder.stop();
  clearInterval(recordingTimer);
  recordButton.disabled = false;
  stopRecordButton.disabled = true;
  recordButton.classList.remove("recording");
  recordButton.setAttribute("aria-label", "開始錄音");
});

discardRecordButton.addEventListener("click", clearSelectedAudio);

analyzeButton.addEventListener("click", async () => {
  const file = selectedAudio;
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
    if (payload.context_override) {
      payload.context = payload.context_override;
    } else {
      try {
        payload.context = await requestSpeechContext(payload.decision.selected_text);
      } catch (contextError) {
        payload.context_error = contextError.message;
      }
    }
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
  if (data.context) applyModelRelationship(data.context);
  else inferRelationship(decision.selected_text);
  $("#audioDuration").textContent = data.audio_duration_seconds == null ? "已審核修正記憶" : `${round(data.audio_duration_seconds)} 秒`;
  $("#processingTime").textContent = `${round(data.dual_pass_seconds)} 秒`;
  $("#baselineConfidence").textContent = round(data.baseline.confidence, 4);
  $("#promptConfidence").textContent = round(data.prompt.confidence, 4);
  $("#decisionReasons").textContent = data.correction_applied ? "已套用人工審核修正" : decision.reasons.join("、");
  $("#baselineText").textContent = data.baseline.text;
  $("#promptText").textContent = data.prompt.text;
  $("#selectedText").textContent = decision.selected_text;
  $("#correctedTranscript").value = decision.selected_text;
  $("#manualText").value = "";
  $("#datasetConsent").checked = false;
  document.querySelectorAll('input[name="candidate"]').forEach((input) => {
    input.checked = input.value === decision.selected_source;
  });
  hide($("#confirmedMessage"));
  hide($("#editTranscriptPanel"));
  renderContextInterpretation(decision.selected_text, decision.reasons, "", data.context);
  const badge = $("#statusBadge");
  if (data.correction_applied) {
    $("#statusTitle").textContent = "已套用人工確認內容";
    badge.textContent = "人工審核";
    badge.classList.remove("warning");
    hide($("#confirmationBox"));
  } else if (decision.needs_confirmation) {
    $("#statusTitle").textContent = "這段語音需要確認";
    badge.textContent = "請確認";
    badge.classList.add("warning");
    show($("#confirmationBox"));
  } else {
    badge.textContent = "理解完成";
    badge.classList.remove("warning");
    hide($("#confirmationBox"));
  }
  show(resultPanel);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function requestSpeechContext(text, extraContext = "") {
  const response = await fetch("/speech/context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      speaker_hint: $("#speakerRole").value || "不確定",
      listener_hint: $("#listenerRole").value || "不確定",
      extra_context: extraContext,
    }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "語境分析失敗");
  return payload;
}

function applyModelRelationship(context) {
  setSelectValue("#speakerRole", context.speaker_role || "不確定");
  setSelectValue("#listenerRole", context.listener_role || "不確定");
  const status = $("#relationStatus");
  const speakerKnown = context.speaker_role && context.speaker_role !== "不確定";
  const listenerKnown = context.listener_role && context.listener_role !== "不確定";
  const relationshipNeedsConfirmation = context.relationship_needs_confirmation ?? context.needs_confirmation;
  const certain = speakerKnown && listenerKnown && !relationshipNeedsConfirmation && Number(context.relationship_confidence) >= 0.65;
  if (certain) status.textContent = `AI 推定 ${Math.round(context.relationship_confidence * 100)}%，可手動調整`;
  else if (speakerKnown || listenerKnown) {
    const known = [speakerKnown ? `說話者：${context.speaker_role}` : "", listenerKnown ? `接收者：${context.listener_role}` : ""].filter(Boolean).join("、");
    status.textContent = `AI 已辨識 ${known}；請確認另一方`;
  } else status.textContent = "AI 資訊不足，請確認人物關係";
  status.classList.toggle("uncertain", !certain);
}

function setSelectValue(id, value) {
  const select = $(id);
  const option = [...select.options].find((item) => item.text === value);
  if (option) select.value = option.value;
}

function contextList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (value == null || value === "") return [];
  return [String(value).trim()].filter(Boolean);
}

function inferRelationship(text) {
  const normalized = text.replace(/[，。！？,.!?\s]/g, "");
  let speaker = "不確定", listener = "不確定", certain = false;
  if (/^(阿嬤|阿公)/.test(normalized)) {
    listener = "長輩";
  } else if (/你食飽未|轉來阮兜|家己挾|莫閣食泡麵|毋免提物件/.test(normalized)) {
    // This caring/inviting phrasing does not identify either person's role.
  } else if (/會議|同事|主管|工作/.test(normalized)) {
    speaker = "同事"; listener = "同事";
  } else if (/媽媽|爸爸/.test(normalized)) {
    listener = "長輩";
  }
  setSelectValue("#speakerRole", speaker); setSelectValue("#listenerRole", listener);
  const status = $("#relationStatus");
  status.textContent = certain ? "系統已推定，可手動調整" : "資訊不足，請確認";
  status.classList.toggle("uncertain", !certain);
}

function renderContextInterpretation(text, reasons = [], extraContext = "", modelContext = null) {
  const speaker = $("#speakerRole").value;
  const listener = $("#listenerRole").value;
  const normalized = text.replace(/[，。！？,.!?\s]/g, "");
  let intent = "這句話主要是在傳達日常資訊；更深一層的意圖仍需依前後文確認。";
  let tags = ["日常對話"];
  let reply = "我知道了，謝謝你告訴我。";
  if (/食飽|吃飯|食飯|食暗頓|食晝/.test(normalized)) {
    intent = "表面上詢問是否吃飯，也可能是在表達關心，或邀請對方一起用餐。";
    tags = ["關心", "問候", "邀請"];
    reply = "好啊，我若還沒吃，等一下就回去。你也要記得吃飯。";
  } else if (/落雨|雨傘|下雨/.test(normalized)) {
    intent = "提醒天氣與外出安全，可能是在關心對方是否準備妥當。";
    tags = ["提醒", "關心"];
    reply = "好，我會記得帶雨傘，你放心。";
  } else if (/轉去|回去|回家|來坐|看妳|看你/.test(normalized)) {
    intent = "說明返家或探望安排，也可能是在回應家人的期待。";
    tags = ["行程", "牽掛", "回應"];
    reply = "好，我知道了。時間確定後再跟你說，你不用擔心。";
  } else if (/毋免|不用|不要|莫/.test(normalized)) {
    intent = "除了字面上的勸阻，也可能是體貼對方、不希望造成負擔。";
    tags = ["體貼", "提醒"];
    reply = "好，我會注意，也謝謝你替我著想。";
  }
  if (modelContext) {
    const modelIntents = contextList(modelContext.possible_intent || modelContext.possible_intents);
    const modelBasis = contextList(modelContext.basis || modelContext.judgment_basis);
    $("#literalMeaning").textContent = modelContext.literal_meaning;
    $("#intentMeaning").textContent = modelIntents.join("；") || "資訊不足，請補充前後文。";
    $("#intentTags").replaceChildren(...modelIntents.map((item) => {
      const tag = document.createElement("span");
      tag.textContent = item.replace(/^可能/, "").slice(0, 12);
      return tag;
    }));
    $("#judgmentBasis").textContent = modelBasis.join("・") || "目前沒有足夠的判斷依據。";
    $("#suggestedReply").textContent = modelContext.suggested_reply;
    return;
  }
  $("#literalMeaning").textContent = text;
  $("#intentMeaning").textContent = intent;
  $("#intentTags").innerHTML = tags.map((tag) => `<span>${tag}</span>`).join("");
  const basis = [`${speaker}對${listener}的對話關係`, "句中的家庭與生活用語"];
  if (extraContext) basis.push(`補充情境：${extraContext}`);
  if (reasons.includes("candidates_disagree")) basis.push("辨識候選不一致，採保守判讀");
  $("#judgmentBasis").textContent = basis.join("・");
  $("#suggestedReply").textContent = reply;
}

$("#confirmButton").addEventListener("click", async () => {
  const manualText = $("#manualText").value.trim();
  const selected = $('input[name="candidate"]:checked');
  let confirmedText = manualText;
  if (!confirmedText && selected && latestResult) confirmedText = latestResult[selected.value].text;
  if (!confirmedText) return alert("請選擇候選內容或輸入修正文字。");
  const button = $("#confirmButton");
  button.disabled = true;
  try {
    const response = await fetch("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio_file: latestResult.filename, audio_sha256: latestResult.audio_sha256, baseline_text: latestResult.baseline.text, prompt_text: latestResult.prompt.text, selected_source: manualText ? "manual" : selected.value, corrected_text: confirmedText, reasons: latestResult.decision.reasons, consent_to_dataset: $("#datasetConsent").checked }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法儲存確認內容");
    $("#confirmedMessage").textContent = `已確認並送交審核：${confirmedText}`;
    show($("#confirmedMessage"));
    $("#selectedText").textContent = confirmedText;
    try {
      latestResult.context = await requestSpeechContext(confirmedText);
      applyModelRelationship(latestResult.context);
      renderContextInterpretation(confirmedText, latestResult.decision.reasons, "", latestResult.context);
    } catch (_) {
      renderContextInterpretation(confirmedText, latestResult.decision.reasons);
    }
  } catch (error) { alert(error.message); }
  finally { button.disabled = false; }
});

$("#contextButton").addEventListener("click", () => $("#contextPanel").classList.toggle("hidden"));
$("#editTranscriptButton").addEventListener("click", () => {
  $("#correctedTranscript").value = $("#selectedText").textContent.trim();
  $("#transcriptDatasetConsent").checked = false;
  hide($("#transcriptSavedMessage"));
  show($("#editTranscriptPanel"));
  $("#editTranscriptPanel").scrollIntoView({ behavior: "smooth", block: "center" });
});
$("#cancelTranscriptEdit").addEventListener("click", () => hide($("#editTranscriptPanel")));
$("#saveTranscriptEdit").addEventListener("click", async () => {
  if (!latestResult?.audio_sha256) return alert("找不到這段音訊的識別碼，請重新辨識後再儲存。");
  const corrected = $("#correctedTranscript").value.trim();
  if (!corrected) return alert("請輸入正確的辨識文字。");
  const saveButton = $("#saveTranscriptEdit");
  saveButton.disabled = true;
  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_file: latestResult.filename || "browser-recording.webm",
        audio_sha256: latestResult.audio_sha256,
        baseline_text: latestResult.baseline.text,
        prompt_text: latestResult.prompt.text,
        selected_source: "manual",
        corrected_text: corrected,
        reasons: ["manual_transcript_correction"],
        consent_to_dataset: $("#transcriptDatasetConsent").checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法儲存辨識修正");
    $("#selectedText").textContent = corrected;
    latestResult.decision.selected_text = corrected;
    latestResult.context = await requestSpeechContext(corrected, $("#extraContext").value.trim());
    applyModelRelationship(latestResult.context);
    renderContextInterpretation(corrected, latestResult.decision.reasons, $("#extraContext").value.trim(), latestResult.context);
    $("#transcriptSavedMessage").textContent = payload.dataset_eligible
      ? "辨識修正已送交審核；核准後可納入資料集。"
      : "辨識修正已送交審核；未授權納入資料集。";
    show($("#transcriptSavedMessage"));
  } catch (error) {
    alert(error.message);
  } finally {
    saveButton.disabled = false;
  }
});
$("#editInsightButton").addEventListener("click", () => {
  $("#editLiteral").value = $("#literalMeaning").textContent;
  $("#editIntent").value = $("#intentMeaning").textContent;
  $("#editBasis").value = $("#judgmentBasis").textContent;
  $("#editReply").value = $("#suggestedReply").textContent;
  $("#contextDatasetConsent").checked = false;
  hide($("#insightSavedMessage"));
  show($("#editInsightPanel"));
  $("#editInsightPanel").scrollIntoView({ behavior: "smooth", block: "center" });
});
$("#cancelInsightEdit").addEventListener("click", () => hide($("#editInsightPanel")));
$("#saveInsightEdit").addEventListener("click", async () => {
  const literal = $("#editLiteral").value.trim();
  const intent = $("#editIntent").value.trim();
  const basis = $("#editBasis").value.trim();
  const reply = $("#editReply").value.trim();
  if (!literal || !intent || !basis || !reply) return alert("四個欄位都需要保留內容；不確定時可填寫「資訊不足」。");
  if (!latestResult?.audio_sha256) return alert("找不到這段音訊的識別碼，請重新辨識後再儲存。");
  const button = $("#saveInsightEdit");
  const humanContext = { literal_meaning: literal, possible_intent: intent, basis, suggested_reply: reply };
  button.disabled = true;
  try {
    const transcription = $("#selectedText").textContent.trim();
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_file: latestResult.filename || "browser-recording.webm",
        audio_sha256: latestResult.audio_sha256,
        baseline_text: transcription,
        prompt_text: transcription,
        selected_source: "manual",
        corrected_text: transcription,
        reasons: [
          "human_context_edit",
          `context_edit_json:${JSON.stringify({ ai_context: latestResult.context || {}, human_context: humanContext })}`,
        ],
        feedback_type: "speech_context",
        ai_context: latestResult.context || {},
        human_context: humanContext,
        consent_to_dataset: $("#contextDatasetConsent").checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法儲存語境修正");
    $("#literalMeaning").textContent = literal;
    $("#intentMeaning").textContent = intent;
    $("#judgmentBasis").textContent = basis;
    $("#suggestedReply").textContent = reply;
    latestResult.human_context_edit = humanContext;
    $("#insightSavedMessage").textContent = payload.dataset_eligible
      ? "修改已送交管理者審核；核准後會套用至相同音檔。"
      : "修改已送交管理者審核；未授權納入資料集。";
    show($("#insightSavedMessage"));
    window.setTimeout(() => hide($("#editInsightPanel")), 1400);
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
});
$("#applyContext").addEventListener("click", async () => {
  if (!latestResult) return;
  const extraContext = $("#extraContext").value.trim();
  try {
    latestResult.context = await requestSpeechContext($("#selectedText").textContent, extraContext);
    applyModelRelationship(latestResult.context);
    renderContextInterpretation($("#selectedText").textContent, latestResult.decision.reasons, extraContext, latestResult.context);
  } catch (error) {
    alert(error.message);
    return;
  }
  hide($("#contextPanel"));
});
$("#resetButton").addEventListener("click", () => {
  clearSelectedAudio();
  latestResult = null;
  hide(resultPanel); hide(errorMessage); hide($("#editTranscriptPanel")); hide($("#editInsightPanel")); hide($("#contextPanel"));
  window.scrollTo({ top: 0, behavior: "smooth" });
});
document.querySelectorAll("#speakerRole,#listenerRole").forEach((select) => select.addEventListener("change", async () => {
  if (!latestResult) return;
  const status = $("#relationStatus");
  status.textContent = "依確認的人物關係重新分析中…";
  try {
    latestResult.context = await requestSpeechContext($("#selectedText").textContent, $("#extraContext").value.trim());
    applyModelRelationship(latestResult.context);
    renderContextInterpretation($("#selectedText").textContent, latestResult.decision.reasons, $("#extraContext").value.trim(), latestResult.context);
  } catch (error) {
    status.textContent = "重新分析失敗，請稍後再試";
    status.classList.add("uncertain");
  }
}));
