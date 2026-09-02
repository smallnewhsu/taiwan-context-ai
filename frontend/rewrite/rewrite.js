const button = document.querySelector("#rewriteButton");
const progress = document.querySelector("#rewriteProgress");
const errorBox = document.querySelector("#rewriteError");
const resultBox = document.querySelector("#rewriteResult");

button.addEventListener("click", async () => {
  const originalText = document.querySelector("#originalText").value.trim();
  if (!originalText) {
    alert("請先輸入想說的話。");
    return;
  }
  button.disabled = true;
  progress.classList.remove("hidden");
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  try {
    const response = await fetch("/expression/rewrite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_text: originalText,
        audience: document.querySelector("#audience").value,
        tone: document.querySelector("#tone").value,
        scenario: document.querySelector("#scenario").value.trim(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "改寫失敗");
    document.querySelector("#rewrittenText").textContent = payload.rewritten_text;
    document.querySelector("#sourceText").textContent = payload.original_text;
    document.querySelector("#meaningSummary").textContent = payload.meaning_summary || "未提供";
    document.querySelector("#modelName").textContent = payload.model;
    document.querySelector("#rewriteTime").textContent = `${payload.processing_seconds.toFixed(2)} 秒`;
    const warning = document.querySelector("#rewriteWarning");
    if (payload.needs_confirmation) {
      warning.textContent = payload.fallback_used
        ? `LLM候選未通過語意保真檢查，已改用保守式安全改寫，請確認後使用。原因：${payload.warnings.join("、")}`
        : `請確認原意是否保留：${payload.warnings.join("、")}`;
      warning.classList.remove("hidden");
    } else {
      warning.classList.add("hidden");
    }
    resultBox.classList.remove("hidden");
    resultBox.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
    progress.classList.add("hidden");
  }
});
