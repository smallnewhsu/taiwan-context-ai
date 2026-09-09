const $ = (selector) => document.querySelector(selector);
const button = $("#rewriteButton");
const progress = $("#rewriteProgress");
const errorBox = $("#rewriteError");
const resultBox = $("#rewriteResult");
let lastPayload = null;

function roleLabel(select) { return select.options[select.selectedIndex].text; }
function updateReminder() { $("#contextReminder").textContent = `訊息較簡短，${roleLabel($("#audience"))}可能擔心你的狀況；系統會保留原意並調整語氣。`; }
$("#audience").addEventListener("change", updateReminder);

async function requestRewrite(tone, scenarioSuffix = "") {
  const scenario = $("#scenario").value.trim();
  const response = await fetch("/expression/rewrite", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ original_text: $("#originalText").value.trim(), audience: $("#audience").value, tone, scenario: [scenario, scenarioSuffix].filter(Boolean).join("；") }) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "改寫失敗");
  return payload;
}

button.addEventListener("click", runRewrite);
async function runRewrite() {
  const originalText = $("#originalText").value.trim();
  if (!originalText) return alert("請先輸入想說的話。");
  button.disabled = true; progress.classList.remove("hidden"); errorBox.classList.add("hidden"); resultBox.classList.add("hidden");
  try {
    const warm = await requestRewrite("warm");
    let taiwanese;
    try { taiwanese = await requestRewrite("warm", "若原意能完整保留，請使用自然的臺灣台語漢字表達；不能安全轉換時保留原句"); }
    catch { taiwanese = warm; }
    lastPayload = warm;
    $("#directText").textContent = originalText;
    $("#rewrittenText").textContent = warm.rewritten_text;
    $("#taiwaneseText").textContent = taiwanese.rewritten_text;
    $("#sourceText").textContent = warm.original_text;
    $("#meaningSummary").textContent = warm.meaning_summary || "保留原句的主要訊息與限制條件。";
    $("#modelName").textContent = warm.model;
    $("#rewriteTime").textContent = `${(warm.processing_seconds + (taiwanese.processing_seconds || 0)).toFixed(2)} 秒`;
    const warnings = [...new Set([...(warm.warnings || []), ...(taiwanese.warnings || [])])];
    if (warm.needs_confirmation || taiwanese.needs_confirmation) { $("#rewriteWarning").textContent = `部分候選使用安全回退，使用前請確認原意。原因：${warnings.join("、")}`; $("#rewriteWarning").classList.remove("hidden"); }
    else $("#rewriteWarning").classList.add("hidden");
    $("#useMessage").classList.add("hidden"); resultBox.classList.remove("hidden"); resultBox.scrollIntoView({behavior:"smooth",block:"start"});
  } catch (error) { errorBox.textContent = error.message; errorBox.classList.remove("hidden"); }
  finally { button.disabled = false; progress.classList.add("hidden"); }
}

$("#clearButton").addEventListener("click", () => { $("#originalText").value=""; $("#scenario").value=""; resultBox.classList.add("hidden"); $("#originalText").focus(); });
$("#regenerateButton").addEventListener("click", runRewrite);
$("#useButton").addEventListener("click", async () => {
  const selected = $('input[name="rewriteStyle"]:checked').value;
  const id = selected === "direct" ? "#directText" : selected === "taiwanese" ? "#taiwaneseText" : "#rewrittenText";
  const text = $(id).textContent;
  try { await navigator.clipboard.writeText(text); $("#useMessage").textContent = "已複製所選版本，可以直接貼到訊息中。"; }
  catch { $("#useMessage").textContent = `已選擇：${text}`; }
  $("#useMessage").classList.remove("hidden");
});
updateReminder();
