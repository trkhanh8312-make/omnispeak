const BASE = window.location.origin;

// ---- Mật khẩu bảo vệ API (đặt cùng giá trị trong Colab Secrets: OMNISPEAK_SECRET) ----
// Hỏi 1 lần đầu trên mỗi trình duyệt, sau đó lưu ở localStorage cho các lần mở lại sau.
let APP_KEY = localStorage.getItem('omnispeak_key') || '';
function ensureKey() {
  if (!APP_KEY) {
    APP_KEY = window.prompt('Nhập mật khẩu truy cập OmniSpeak:') || '';
    localStorage.setItem('omnispeak_key', APP_KEY);
  }
}
async function apiFetch(url, options = {}) {
  ensureKey();
  const doFetch = () => fetch(url, { ...options, headers: { ...(options.headers || {}), 'X-Omnispeak-Key': APP_KEY } });
  let res = await doFetch();
  if (res.status === 401) {
    alert('Sai mật khẩu hoặc mật khẩu đã đổi — nhập lại.');
    localStorage.removeItem('omnispeak_key');
    APP_KEY = window.prompt('Nhập mật khẩu truy cập OmniSpeak:') || '';
    localStorage.setItem('omnispeak_key', APP_KEY);
    res = await doFetch();
  }
  return res;
}

let mediaRecorder, recordedChunks = [], recordedBlob = null, isRecording = false;
let historyItems = [];
let allProfiles = [];
let selectedProfileId = "";
let selectedProfileName = "Mặc định";
let lastGeneratedKey = null;  // text + '||' + profileId của lần đọc gần nhất thành công

// ---- Giới hạn độ dài văn bản ----
const SOFT_WORD_LIMIT = 2000;  // vẫn tạo được, chỉ cảnh báo sẽ xử lý theo từng đoạn + mất thời gian
const HARD_WORD_LIMIT = 3000;  // vượt mức này thì chặn, yêu cầu rút gọn
const MAX_PROFILES = 20;       // khớp với MAX_PROFILES ở backend.py — chỉ để hiển thị, backend là nơi chặn thật

// ---- Theme (mặc định: tối) ----
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved !== 'light') document.body.classList.add('dark');
  updateThemeIcon();
}
function toggleTheme() {
  document.body.classList.toggle('dark');
  localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light');
  updateThemeIcon();
}
function updateThemeIcon() {
  document.getElementById('themeToggle').textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
}

// ---- Tab switching (sidebar) ----
document.querySelectorAll('.navitem').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.navitem').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('tab-' + el.dataset.tab).classList.add('active');
    if (el.dataset.tab === 'library' || el.dataset.tab === 'tts') loadVoices();
  });
});

// ---- Right-panel segmented tabs (Giọng nói / Lịch sử) ----
function switchRightTab(tab) {
  document.getElementById('segVoices').classList.toggle('active', tab === 'voices');
  document.getElementById('segHistory').classList.toggle('active', tab === 'history');
  document.getElementById('rightPanel-voices').style.display = tab === 'voices' ? 'flex' : 'none';
  document.getElementById('rightPanel-history').style.display = tab === 'history' ? 'flex' : 'none';
}

function setStatus(id, msg, cls) {
  const el = document.getElementById(id);
  el.textContent = msg || '';
  el.className = 'status' + (cls ? ' ' + cls : '');
}

// ---- Escape text người dùng nhập trước khi chèn vào innerHTML (chống XSS) ----
function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

// ---- Lấy thông báo lỗi dễ đọc từ response lỗi của FastAPI (JSON {"detail": "..."}) ----
async function extractErrorMessage(response) {
  try {
    const data = await response.json();
    if (data && data.detail) return data.detail;
  } catch (e) { /* không phải JSON */ }
  return 'HTTP ' + response.status;
}

// ---- Character / word counters + cảnh báo độ dài ----
function updateCounts() {
  const text = document.getElementById('ttsText').value;
  document.getElementById('charCount').textContent = text.length + ' ký tự';
  const words = text.trim().length ? text.trim().split(/\s+/).length : 0;
  document.getElementById('wordCount').textContent = words + ' từ';

  const warnEl = document.getElementById('lengthWarning');
  const genBtn = document.getElementById('genBtn');
  if (words > HARD_WORD_LIMIT) {
    warnEl.textContent = `Văn bản vượt quá giới hạn ${HARD_WORD_LIMIT} từ (hiện tại: ${words} từ) — vui lòng rút ngắn bớt.`;
    warnEl.className = 'status err';
    genBtn.disabled = true;
    genBtn.textContent = '▷ Tạo giọng đọc';
    genBtn.title = '';
    return;
  } else if (words > SOFT_WORD_LIMIT) {
    warnEl.textContent = `Văn bản khá dài (${words} từ) — sẽ được xử lý theo từng đoạn nhỏ và có thể mất vài phút.`;
    warnEl.className = 'status warn';
  } else {
    warnEl.textContent = '';
    warnEl.className = 'status';
  }
  updateGenButtonState();
}

// ---- Chặn đọc lại đoạn văn + giọng chưa đổi so với lần đọc gần nhất ----
function updateGenButtonState() {
  const text = document.getElementById('ttsText').value.trim();
  const genBtn = document.getElementById('genBtn');
  const key = text + '||' + selectedProfileId;

  if (text && lastGeneratedKey !== null && key === lastGeneratedKey) {
    genBtn.disabled = true;
    genBtn.textContent = '✓ Đã đọc xong';
    genBtn.title = 'Đoạn văn này đã đọc rồi bằng giọng hiện tại — sửa nội dung hoặc đổi giọng để đọc lại.';
  } else {
    genBtn.disabled = false;
    genBtn.textContent = '▷ Tạo giọng đọc';
    genBtn.title = '';
  }
}

// ---- Filename helper: "voice-<ngay-gio>.wav" ----
function makeFilename(text, ext) {
  ext = ext || 'wav';
  const now = new Date();
  const pad = n => String(n).padStart(2, '0');
  const stamp = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `voice-${stamp}.${ext}`;
}

// ---- Load saved voice profiles ----
async function loadVoices() {
  try {
    const r = await apiFetch(BASE + '/profiles');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    allProfiles = await r.json();

    if (selectedProfileId && !allProfiles.some(p => p.id === selectedProfileId)) {
      selectedProfileId = "";
      selectedProfileName = "Mặc định";
      document.getElementById('selectedVoiceName').textContent = selectedProfileName;
      updateGenButtonState();
    }

    renderVoicePicker();

    const listEl = document.getElementById('voiceList');
    if (allProfiles.length === 0) {
      listEl.innerHTML = '<div class="empty">Chưa có giọng nào — sang tab "Giọng nói của bạn" để tạo giọng đầu tiên.</div>';
    } else {
      listEl.innerHTML = `<div class="counts" style="margin-bottom:10px">${allProfiles.length}/${MAX_PROFILES} giọng đã lưu</div>`;
      allProfiles.forEach(p => {
        const div = document.createElement('div');
        div.className = 'voice-item';
        div.innerHTML = `<div style="display:flex;align-items:center;gap:10px">
                            <button class="preview-btn" id="prevbtn-${p.id}" onclick="previewVoice('${p.id}', this)" title="Nghe thử">▷</button>
                            <div><div class="name">${escapeHtml(p.name)}</div><div class="meta">${escapeHtml(p.kind || 'clone')}</div></div>
                          </div>
                          <div class="actions">
                            <button class="btn secondary" onclick="useVoice('${p.id}')">Dùng giọng này</button>
                            <button class="btn danger" onclick="deleteVoice('${p.id}', this)">Xoá</button>
                          </div>`;
        listEl.appendChild(div);
      });
    }
  } catch (e) {
    console.error(e);
  }
}

// ---- Voice picker (right panel of "Đọc văn bản") ----
function renderVoicePicker() {
  const q = (document.getElementById('voiceSearch').value || '').trim().toLowerCase();
  const listEl = document.getElementById('voicePickerList');
  listEl.innerHTML = '';

  const items = [{ id: '', name: 'Mặc định (tự động)', kind: 'system' }, ...allProfiles];
  const filtered = items.filter(p => p.name.toLowerCase().includes(q));

  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="empty">Không tìm thấy giọng nào.</div>';
    return;
  }

  filtered.forEach(p => {
    const row = document.createElement('div');
    row.className = 'voice-row' + (p.id === selectedProfileId ? ' selected' : '');
    row.onclick = () => selectVoice(p.id, p.name);
    const playIcHtml = p.id
      ? `<div class="play-ic" id="pickpic-${p.id}" onclick="previewVoice('${p.id}', this, event)">▷</div>`
      : `<div class="play-ic">•</div>`;
    row.innerHTML = `${playIcHtml}
                      <div><div class="vname">${escapeHtml(p.name)}</div>
                      <div class="vmeta">${p.id ? 'Giọng do bạn tự tạo' : 'Giọng mặc định của hệ thống'}</div></div>`;
    listEl.appendChild(row);
  });
}

// ---- Nghe thử giọng đã lưu (dùng chung 1 thẻ audio ẩn) ----
const libPreviewAudio = document.getElementById('libPreviewAudio');
let previewingId = null;

async function previewVoice(id, btnEl, event) {
  if (event) event.stopPropagation();
  const isSameAndPlaying = previewingId === id && !libPreviewAudio.paused;
  document.querySelectorAll('.preview-btn.playing, .play-ic.playing').forEach(el => el.classList.remove('playing'));
  libPreviewAudio.pause();

  if (isSameAndPlaying) { previewingId = null; return; }

  previewingId = id;
  try {
    const r = await apiFetch(BASE + '/profiles/' + id + '/preview');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const blob = await r.blob();
    libPreviewAudio.src = URL.createObjectURL(blob);
    await libPreviewAudio.play();
    if (btnEl) btnEl.classList.add('playing');
  } catch (e) {
    previewingId = null;
    alert('Giọng này chưa có bản nghe thử (được tạo trước khi có tính năng này).');
  }
}
libPreviewAudio.addEventListener('ended', () => {
  document.querySelectorAll('.preview-btn.playing, .play-ic.playing').forEach(el => el.classList.remove('playing'));
  previewingId = null;
});

function selectVoice(id, name) {
  selectedProfileId = id;
  selectedProfileName = name;
  document.getElementById('selectedVoiceName').textContent = name;
  renderVoicePicker();
  updateGenButtonState();
}

function useVoice(id) {
  const p = allProfiles.find(x => x.id === id);
  document.querySelector('.navitem[data-tab="tts"]').click();
  selectVoice(id, p ? p.name : id);
  switchRightTab('voices');
}

async function deleteVoice(id, btnEl) {
  if (!confirm('Xoá giọng nói này khỏi thư viện?')) return;
  btnEl.disabled = true;
  try {
    const r = await apiFetch(BASE + '/profiles/' + id, { method: 'DELETE' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    loadVoices();
  } catch (e) {
    alert('Không xoá được (backend có thể chưa hỗ trợ xoá qua API): ' + e.message);
    btnEl.disabled = false;
  }
}

// ---- Recording ----
async function toggleRecord() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
      mediaRecorder.onstop = () => {
        recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
        const preview = document.getElementById('refPreview');
        preview.src = URL.createObjectURL(recordedBlob);
        preview.style.display = 'block';
        document.getElementById('saveVoiceBtn').disabled = false;
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      isRecording = true;
      document.getElementById('recBtn').innerHTML = '<span class="rec-dot"></span>Dừng ghi âm';
      setStatus('recStatus', 'Đang ghi âm... đọc to rõ khoảng 5-10 giây.');
    } catch (e) {
      setStatus('recStatus', 'Không truy cập được microphone: ' + e.message, 'err');
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    document.getElementById('recBtn').innerHTML = '🎙️ Bắt đầu ghi âm';
    setStatus('recStatus', 'Đã ghi âm xong — nghe lại bên dưới rồi bấm Lưu.', 'ok');
  }
}
function onFileChosen() {
  const f = document.getElementById('fileInput').files[0];
  if (!f) return;
  recordedBlob = f;
  const preview = document.getElementById('refPreview');
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
  document.getElementById('saveVoiceBtn').disabled = false;
  setStatus('recStatus', 'Đã chọn file: ' + f.name, 'ok');
}

async function saveVoiceProfile() {
  const name = document.getElementById('voiceName').value.trim();
  if (!name) { setStatus('voiceStatus', 'Nhập tên giọng nói trước đã.', 'err'); return; }
  if (!recordedBlob) { setStatus('voiceStatus', 'Chưa có audio mẫu.', 'err'); return; }
  const btn = document.getElementById('saveVoiceBtn');
  btn.disabled = true;
  setStatus('voiceStatus', 'Đang lưu giọng nói...');
  try {
    const fd = new FormData();
    fd.append('name', name);
    fd.append('kind', 'clone');
    fd.append('ref_audio', recordedBlob, 'reference.wav');
    const r = await apiFetch(BASE + '/profiles', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await extractErrorMessage(r));
    setStatus('voiceStatus', 'Đã lưu giọng "' + name + '" vào thư viện.', 'ok');
    document.getElementById('voiceName').value = '';
    recordedBlob = null;
    document.getElementById('refPreview').style.display = 'none';
    loadVoices();
  } catch (e) {
    setStatus('voiceStatus', 'Lưu thất bại: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

// ---- TTS generation (job + polling — xử lý theo từng đoạn cho văn bản dài) ----
function sleep(ms) { return new Promise(res => setTimeout(res, ms)); }

async function generateTTS() {
  const text = document.getElementById('ttsText').value.trim();
  if (!text) { setStatus('ttsStatus', 'Nhập văn bản trước đã.', 'err'); return; }
  const words = text.split(/\s+/).length;
  if (words > HARD_WORD_LIMIT) { setStatus('ttsStatus', `Văn bản vượt quá ${HARD_WORD_LIMIT} từ.`, 'err'); return; }

  const btn = document.getElementById('genBtn');
  btn.disabled = true;
  setStatus('ttsStatus', 'Đang gửi văn bản...');
  try {
    const fd = new FormData();
    fd.append('text', text);
    if (selectedProfileId) fd.append('profile_id', selectedProfileId);
    const r = await apiFetch(BASE + '/generate', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await extractErrorMessage(r));
    const { job_id, total_chunks } = await r.json();

    // Poll trạng thái tới khi xong
    let status = null;
    while (true) {
      const sr = await apiFetch(BASE + '/generate/' + job_id + '/status');
      if (!sr.ok) throw new Error('HTTP ' + sr.status);
      status = await sr.json();
      if (status.status === 'done' || status.status === 'error') break;
      setStatus('ttsStatus', total_chunks > 1
        ? `Đang tạo giọng nói... (đoạn ${status.done}/${status.total})`
        : 'Đang tạo giọng nói...');
      await sleep(900);
    }
    if (status.status === 'error') throw new Error(status.error || 'Lỗi không xác định');

    const ar = await apiFetch(BASE + '/generate/' + job_id + '/audio');
    if (!ar.ok) throw new Error('HTTP ' + ar.status);
    const blob = await ar.blob();
    const url = URL.createObjectURL(blob);
    const filename = makeFilename(text);
    const genTime = ar.headers.get('X-Gen-Time');

    setStatus('ttsStatus', genTime ? `Xong trong ${genTime}s.` : 'Xong.', 'ok');
    showPlayer(url, filename, selectedProfileName, genTime);
    addHistory(text, selectedProfileName, url, filename, genTime);

    // Đánh dấu đoạn văn + giọng này đã đọc — chặn đọc lại tới khi đổi nội dung hoặc giọng
    lastGeneratedKey = text + '||' + selectedProfileId;
  } catch (e) {
    setStatus('ttsStatus', 'Tạo thất bại: ' + e.message, 'err');
  } finally {
    updateGenButtonState();
  }
}

function addHistory(text, voiceName, url, filename, genTime) {
  historyItems.unshift({ text, voiceName, url, filename, genTime, time: new Date().toLocaleTimeString('vi-VN') });
  renderHistory();
}
function renderHistory() {
  const fullEl = document.getElementById('historyList');
  const panelEl = document.getElementById('historyPanelList');

  if (historyItems.length === 0) {
    fullEl.innerHTML = '<div class="empty">Chưa có lịch sử nào.</div>';
    panelEl.innerHTML = '<div class="empty">Chưa có lịch sử nào.</div>';
    return;
  }

  fullEl.innerHTML = '';
  historyItems.forEach(h => {
    const div = document.createElement('div');
    div.className = 'hist-item';
    div.innerHTML = `<div class="text">${escapeHtml(h.text.length > 120 ? h.text.slice(0,120)+'…' : h.text)}</div>
                      <div class="meta">${escapeHtml(h.voiceName)} · ${escapeHtml(h.time)}</div>
                      <audio controls src="${h.url}"></audio>
                      <div class="row">
                        <a href="${h.url}" download="${h.filename}" class="btn secondary">⬇️ Tải xuống</a>
                      </div>`;
    fullEl.appendChild(div);
  });

  panelEl.innerHTML = '';
  historyItems.forEach(h => {
    const row = document.createElement('div');
    row.className = 'hist-row';
    row.onclick = () => showPlayer(h.url, h.filename, h.voiceName, h.genTime);
    row.innerHTML = `<div class="htext">${escapeHtml(h.text.length > 40 ? h.text.slice(0,40)+'…' : h.text)}</div>
                      <div class="hmeta">${escapeHtml(h.voiceName)} · ${escapeHtml(h.time)}</div>`;
    panelEl.appendChild(row);
  });
}

// ---- Bottom player bar ----
const playerAudio = document.getElementById('playerAudio');
let currentSpeed = 1;

function fmtTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function showPlayer(url, filename, voiceName, genTime) {
  playerAudio.src = url;
  playerAudio.playbackRate = currentSpeed;
  document.getElementById('playerName').textContent = filename;
  document.getElementById('playerMeta').textContent = voiceName + (genTime ? ' · ' + genTime + 's' : '');
  document.getElementById('playerDownload').href = url;
  document.getElementById('playerDownload').download = filename;
  document.getElementById('playerBar').style.display = 'flex';
  playerAudio.play();
}

function togglePlayPause() {
  if (!playerAudio.src) return;
  if (playerAudio.paused) playerAudio.play(); else playerAudio.pause();
}
function stopPlayer() {
  if (!playerAudio.src) return;
  playerAudio.pause();
  playerAudio.currentTime = 0;
}
function cycleSpeed() {
  const speeds = [1, 1.25, 1.5, 2, 0.75];
  const idx = speeds.indexOf(currentSpeed);
  currentSpeed = speeds[(idx + 1) % speeds.length];
  playerAudio.playbackRate = currentSpeed;
  document.getElementById('speedBtn').textContent = currentSpeed + 'x';
}
function toggleMute() {
  playerAudio.muted = !playerAudio.muted;
  document.getElementById('muteBtn').textContent = playerAudio.muted ? '🔇' : '🔊';
}
function seekPlayer() {
  if (!playerAudio.duration) return;
  const pct = document.getElementById('playerSeek').value;
  playerAudio.currentTime = (pct / 100) * playerAudio.duration;
}
function closePlayer() {
  playerAudio.pause();
  document.getElementById('playerBar').style.display = 'none';
}

playerAudio.addEventListener('play', () => { document.getElementById('playPauseBtn').textContent = '⏸'; });
playerAudio.addEventListener('pause', () => { document.getElementById('playPauseBtn').textContent = '▷'; });
playerAudio.addEventListener('timeupdate', () => {
  document.getElementById('playerTime').textContent = fmtTime(playerAudio.currentTime) + ' / ' + fmtTime(playerAudio.duration);
  const seek = document.getElementById('playerSeek');
  if (playerAudio.duration) {
    const pct = (playerAudio.currentTime / playerAudio.duration) * 100;
    seek.value = pct;
    seek.style.setProperty('--pct', pct + '%');
  }
});
playerAudio.addEventListener('loadedmetadata', () => {
  document.getElementById('playerTime').textContent = fmtTime(0) + ' / ' + fmtTime(playerAudio.duration);
});

// ---- Giữ phiên sống + báo khi mất kết nối backend ----
let backendUp = true;
async function heartbeat() {
  try {
    const r = await fetch(BASE + '/health', { cache: 'no-store' });
    if (!r.ok) throw new Error('bad status');
    if (!backendUp) { loadVoices(); }  // vừa kết nối lại -> nạp lại dữ liệu
    backendUp = true;
    document.getElementById('connBanner').style.display = 'none';
  } catch (e) {
    backendUp = false;
    document.getElementById('connBanner').style.display = 'block';
  }
}
setInterval(heartbeat, 60000);  // mỗi 60s — vừa giữ tab "có hoạt động", vừa dò kết nối

initTheme();
updateCounts();
loadVoices();
heartbeat();
