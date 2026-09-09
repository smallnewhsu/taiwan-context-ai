const list = document.querySelector("#reviewList");
const status = document.querySelector("#reviewStatus");
const errorBox = document.querySelector("#reviewError");

const escapeText = (value) => String(value ?? "");

function contextPayload(item) {
  if (item.ai_context && item.human_context) {
    return { ai_context: item.ai_context, human_context: item.human_context };
  }
  const encoded = (item.reasons || []).find((reason) => String(reason).startsWith("context_edit_json:"));
  if (!encoded) return null;
  try { return JSON.parse(String(encoded).slice("context_edit_json:".length)); }
  catch { return null; }
}

async function loadPending() {
  status.classList.remove("hidden");
  errorBox.classList.add("hidden");
  try {
    const response = await fetch("/feedback/pending");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法載入待審核資料");
    const consentedOnly = new URLSearchParams(location.search).get("consented") === "1";
    render(consentedOnly ? payload.items.filter((item) => item.consent_to_dataset) : payload.items);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove("hidden");
  } finally {
    status.classList.add("hidden");
  }
}

function textRow(label, value, corrected = false) {
  const row = document.createElement("div");
  row.className = `text-row${corrected ? " corrected" : ""}`;
  const small = document.createElement("small");
  small.textContent = label;
  const text = document.createElement("strong");
  text.textContent = escapeText(value);
  row.append(small, text);
  return row;
}

function render(items) {
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "panel empty";
    empty.textContent = "目前沒有待審核資料。";
    list.append(empty);
    return;
  }

  items.forEach((item) => {
    const savedContext = contextPayload(item);
    const card = document.createElement("article");
    card.className = "review-card";
    card.dataset.feedbackId = item.feedback_id;

    const meta = document.createElement("div");
    meta.className = "review-meta";
    [item.audio_file, item.selected_source, item.consent_to_dataset ? "已同意資料使用" : "未同意資料使用"].forEach((value) => {
      const span = document.createElement("span");
      span.textContent = value;
      meta.append(span);
    });

    const note = document.createElement("textarea");
    note.className = "review-note";
    note.rows = 2;
    note.placeholder = "審查備註（選填）";

    const actions = document.createElement("div");
    actions.className = "review-actions";
    const approve = document.createElement("button");
    approve.textContent = "核准";
    approve.addEventListener("click", () => review(item.feedback_id, "approve", note.value, card));
    const reject = document.createElement("button");
    reject.className = "reject";
    reject.textContent = "駁回";
    reject.addEventListener("click", () => review(item.feedback_id, "reject", note.value, card));
    actions.append(approve, reject);

    const contentRows = (item.feedback_type === "speech_context" || savedContext)
      ? [
          textRow("確認逐字稿", item.corrected_text),
          textRow("AI 字面意思", savedContext?.ai_context?.literal_meaning || item.ai_context?.literal_meaning || "未提供"),
          textRow("修改後字面意思", savedContext?.human_context?.literal_meaning || item.human_context?.literal_meaning, true),
          textRow("AI 可能意境", (savedContext?.ai_context?.possible_intents || item.ai_context?.possible_intents || []).join("；") || savedContext?.ai_context?.possible_intent || "未提供"),
          textRow("修改後可能意境", savedContext?.human_context?.possible_intent || item.human_context?.possible_intent, true),
          textRow("AI 判斷依據", (savedContext?.ai_context?.basis || item.ai_context?.basis || []).join("・") || "未提供"),
          textRow("修改後判斷依據", savedContext?.human_context?.basis || item.human_context?.basis, true),
          textRow("AI 建議回應", savedContext?.ai_context?.suggested_reply || item.ai_context?.suggested_reply || "未提供"),
          textRow("修改後建議回應", savedContext?.human_context?.suggested_reply || item.human_context?.suggested_reply, true),
        ]
      : [
          textRow("原始辨識", item.baseline_text),
          textRow("提示辨識", item.prompt_text),
          textRow("使用者修正", item.corrected_text, true),
        ];
    if (item.feedback_type === "speech_context" || savedContext) {
      const type = document.createElement("span");
      type.textContent = "語境判斷修正";
      meta.prepend(type);
    }
    card.append(meta, ...contentRows, note, actions);
    list.append(card);
  });
}

async function review(feedbackId, decision, note, card) {
  card.querySelectorAll("button").forEach((button) => button.disabled = true);
  try {
    const response = await fetch(`/feedback/${feedbackId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reviewer_note: note }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "審查失敗");
    card.remove();
    if (!list.children.length) render([]);
  } catch (error) {
    alert(error.message);
    card.querySelectorAll("button").forEach((button) => button.disabled = false);
  }
}

loadPending();
