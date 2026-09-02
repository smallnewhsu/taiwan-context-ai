const list = document.querySelector("#reviewList");
const status = document.querySelector("#reviewStatus");
const errorBox = document.querySelector("#reviewError");

const escapeText = (value) => String(value ?? "");

async function loadPending() {
  status.classList.remove("hidden");
  errorBox.classList.add("hidden");
  try {
    const response = await fetch("/feedback/pending");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "無法載入待審核資料");
    render(payload.items);
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

    card.append(
      meta,
      textRow("原始辨識", item.baseline_text),
      textRow("提示辨識", item.prompt_text),
      textRow("使用者修正", item.corrected_text, true),
      note,
      actions
    );
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
