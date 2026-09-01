const $ = (id) => document.getElementById(id);

const uploadMsg = $("upload-msg");
const listaArquivos = $("lista-arquivos");
const fileCount = $("file-count");
const btnIndexar = $("btn-indexar");
const btnIndexarText = $("btn-indexar-text");
const statusText = $("status-text");
const dotIndex = $("dot-index");
const badgeLlm = $("badge-llm");
const messagesEl = $("messages");
const welcomeEl = $("welcome");
const inputPergunta = $("pergunta");
const btnEnviar = $("btn-enviar");
const dropzone = $("dropzone");
const arquivoInput = $("arquivo");

let statusAtual = { indexado: false, llm_configurada: false, arquivos: [] };
let enviando = false;

function escapar(texto) {
  return texto
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function parseResposta(res) {
  const texto = await res.text();
  try {
    return { ok: res.ok, data: JSON.parse(texto) };
  } catch {
    throw new Error(texto || `Erro ${res.status}`);
  }
}

function setFeedback(el, msg, tipo = "") {
  el.textContent = msg;
  el.className = `feedback ${tipo}`;
}

function atualizarUI() {
  const { indexado, llm_configurada, arquivos, documentos, chunks } = statusAtual;
  const temArquivos = arquivos.length > 0;

  fileCount.textContent = arquivos.length;
  btnIndexar.disabled = !temArquivos || btnIndexarText.textContent === "Indexando...";

  if (indexado) {
    dotIndex.className = "dot ready";
    statusText.textContent = `${documentos} doc · ${chunks} chunks indexados`;
  } else if (temArquivos) {
    dotIndex.className = "dot warning";
    statusText.textContent = `${arquivos.length} arquivo(s) — clique em indexar`;
  } else {
    dotIndex.className = "dot";
    statusText.textContent = "Aguardando documentos";
  }

  if (llm_configurada) {
    badgeLlm.textContent = "Gemini";
    badgeLlm.className = "pill ok";
  } else {
    badgeLlm.textContent = "Sem API key";
    badgeLlm.className = "pill erro";
  }

  const podePerguntar = indexado && llm_configurada && !enviando;
  inputPergunta.disabled = !podePerguntar;
  btnEnviar.disabled = !podePerguntar;

  if (!arquivos.length) {
    listaArquivos.innerHTML = '<li class="empty">Nenhum documento</li>';
  } else {
    listaArquivos.innerHTML = arquivos.map((n) => `<li>${escapar(n)}</li>`).join("");
  }
}

async function carregarStatus() {
  const res = await fetch("/api/status");
  const { data } = await parseResposta(res);
  statusAtual = data;
  atualizarUI();
}

async function enviarArquivo(file) {
  const formData = new FormData();
  formData.append("arquivo", file);

  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const { ok, data } = await parseResposta(res);
  if (!ok) throw new Error(data.detail || "Erro no upload");

  setFeedback(uploadMsg, `Adicionado: ${data.arquivo}`, "ok");
  await carregarStatus();
}

dropzone.addEventListener("click", () => arquivoInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  try {
    await enviarArquivo(file);
  } catch (err) {
    setFeedback(uploadMsg, err.message, "erro");
  }
});

arquivoInput.addEventListener("change", async () => {
  const file = arquivoInput.files[0];
  if (!file) return;
  try {
    await enviarArquivo(file);
  } catch (err) {
    setFeedback(uploadMsg, err.message, "erro");
  }
  arquivoInput.value = "";
});

btnIndexar.addEventListener("click", async () => {
  btnIndexar.disabled = true;
  btnIndexarText.textContent = "Indexando...";

  try {
    const res = await fetch("/api/indexar", { method: "POST" });
    const { ok, data } = await parseResposta(res);
    if (!ok) throw new Error(data.detail || "Erro ao indexar");
    await carregarStatus();
    setFeedback(uploadMsg, `Pronto — ${data.chunks} chunks`, "ok");
  } catch (err) {
    setFeedback(uploadMsg, err.message, "erro");
  } finally {
    btnIndexarText.textContent = "Indexar documentos";
    atualizarUI();
  }
});

function esconderWelcome() {
  if (welcomeEl) welcomeEl.remove();
}

function criarMsgUsuario(texto) {
  esconderWelcome();
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="msg-avatar">Você</div>
    <div class="msg-body">
      <div class="msg-bubble">${escapar(texto)}</div>
    </div>`;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function criarMsgAssistenteLoading() {
  esconderWelcome();
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="msg-label">Resposta do Gemini</div>
      <div class="msg-bubble loading" id="loading-bubble">Buscando nos documentos...</div>
    </div>`;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  setTimeout(() => {
    const el = row.querySelector("#loading-bubble");
    if (el?.classList.contains("loading")) {
      el.textContent = "Gerando resposta com Gemini...";
    }
  }, 1500);

  return row;
}

function renderFontes(container, resultados) {
  if (!resultados.length) return;

  const wrap = document.createElement("div");
  wrap.className = "sources";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sources-toggle";
  btn.innerHTML = `📎 ${resultados.length} fonte(s) usadas ▾`;

  const list = document.createElement("div");
  list.className = "sources-list";
  list.hidden = true;

  btn.addEventListener("click", () => {
    list.hidden = !list.hidden;
    btn.innerHTML = list.hidden
      ? `📎 ${resultados.length} fonte(s) usadas ▾`
      : `📎 ${resultados.length} fonte(s) usadas ▴`;
  });

  list.innerHTML = resultados
    .map(
      (r, i) => `
      <div class="source-card">
        <div class="source-meta">
          <span>${escapar(r.fonte)} · chunk ${r.indice + 1}</span>
          <span class="source-score">${(r.similaridade * 100).toFixed(0)}% match</span>
        </div>
        <div class="source-text">${escapar(r.texto)}</div>
      </div>`
    )
    .join("");

  wrap.append(btn, list);
  container.appendChild(wrap);
}

async function perguntar(texto) {
  if (!texto.trim() || enviando) return;

  enviando = true;
  atualizarUI();
  criarMsgUsuario(texto);
  inputPergunta.value = "";

  const loadingRow = criarMsgAssistenteLoading();
  const bubble = loadingRow.querySelector(".msg-bubble");

  try {
    const res = await fetch("/api/perguntar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta: texto }),
    });
    const { ok, data } = await parseResposta(res);
    if (!ok) throw new Error(data.detail || "Erro na resposta");

    if (!data.resposta) {
      throw new Error("O Gemini não retornou resposta. Verifique GOOGLE_API_KEY no .env");
    }

    bubble.className = "msg-bubble assistant-answer";
    bubble.textContent = data.resposta;
    renderFontes(loadingRow.querySelector(".msg-body"), data.resultados);
  } catch (err) {
    bubble.className = "msg-bubble";
    bubble.style.color = "#dc2626";
    bubble.textContent = err.message;
  } finally {
    enviando = false;
    atualizarUI();
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

$("form-pergunta").addEventListener("submit", (e) => {
  e.preventDefault();
  perguntar(inputPergunta.value);
});

inputPergunta.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("form-pergunta").requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    inputPergunta.value = btn.textContent;
    inputPergunta.focus();
  });
});

carregarStatus();
