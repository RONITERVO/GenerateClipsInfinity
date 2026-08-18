/**
 * ============================================================================
 * YOUTUBE NATIVE THEATER CONTROLLER (theater.js)
 * ============================================================================
 * Handles YouTube watch-page player controls, search bar prompt engine,
 * sidebar recommendations / playlist, live chat steering, and settings modal.
 * ============================================================================
 */

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  let activeId = null;
  let state = null;
  let pollTimer = null;
  let currentIndex = -1;
  let front = 'A';
  let playbackStarted = false;
  let userMuted = false;
  let isCcEnabled = true;
  let lastSegmentCount = 0;
  let directiveRequestInFlight = false;
  let subtitleManager = null;
  let activeSidebarTab = 'scenes'; // 'scenes', 'chat', 'saved'

  const players = {
    A: $('playerA'),
    B: $('playerB'),
  };

  function toast(text) {
    const el = $('toast');
    if (!el) return;
    el.textContent = text;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 2800);
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s ?? '';
    return d.innerHTML;
  }

  function fmt(seconds) {
    seconds = Math.max(0, Math.round(seconds || 0));
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  function mediaUrl(path) {
    return `/api/video?path=${encodeURIComponent(path)}`;
  }

  // --------------------------------------------------------------------------
  // YouTube Search Bar Story Submission
  // --------------------------------------------------------------------------
  async function submitStoryPrompt(e) {
    if (e) e.preventDefault();
    const promptInput = $('ytSearchInput');
    const prompt = (promptInput?.value || '').trim();

    if (!prompt) {
      toast('Enter a story idea or topic in the search bar');
      return;
    }

    const mode = $('modeSelect')?.value || 'edutainment';
    const audience = $('audienceSelect')?.value || 'family';
    const language = $('languageSelect')?.value || 'en';
    const translation_language = $('transLanguageSelect')?.value || '';
    const voice = $('voiceSelect')?.value || 'M1';
    const learning_focus = mode === 'dream' ? '' : ($('learningInput')?.value || '').trim();

    const quality_settings = {
      width: Number($('qualityWidth')?.value || 480),
      height: Number($('qualityHeight')?.value || 272),
      frames: Number($('qualityFrames')?.value || 81),
      fps: Number($('qualityFps')?.value || 16),
      min_words: Number($('minWords')?.value || 80),
      max_words: Number($('maxWords')?.value || 110),
      max_slow: Number($('maxSlow')?.value || 8.0),
    };

    const payload = {
      prompt,
      learning_focus,
      mode,
      audience,
      language,
      translation_language,
      voice,
      quality_settings,
      context_compaction_scenes: Number($('compactionScenes')?.value || 30),
      seed: Number($('seedInput')?.value || -1),
    };

    $('playerWaiting').style.display = 'grid';
    $('waitTitle').textContent = 'Starting endless stream';
    $('waitText').textContent = 'Gemma builds opening buffer and hands RTX GPU to Wan text-to-video.';

    try {
      const r = await fetch('/api/theater', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not start story');

      currentIndex = -1;
      playbackStarted = false;
      lastSegmentCount = 0;
      renderState(data);
      loadRecent();
    } catch (err) {
      toast(err.message);
      $('playerWaiting').style.display = 'none';
    }
  }

  // --------------------------------------------------------------------------
  // YouTube Video Player Controls
  // --------------------------------------------------------------------------
  function togglePlayPause() {
    const player = players[front];
    if (player.paused) {
      player.play().catch(() => {});
      $('btnPlayPause').innerHTML = `<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
      $('ytPlayerContainer').classList.remove('paused');
    } else {
      player.pause();
      $('btnPlayPause').innerHTML = `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
      $('ytPlayerContainer').classList.add('paused');
    }
  }

  function toggleMute() {
    userMuted = !userMuted;
    players.A.muted = userMuted;
    players.B.muted = userMuted;
    $('btnMute').innerHTML = userMuted
      ? `<svg viewBox="0 0 24 24"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>`
      : `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>`;
  }

  function toggleCc() {
    isCcEnabled = !isCcEnabled;
    const captionEl = $('ytCaptionWindow');
    if (captionEl) captionEl.style.display = isCcEnabled ? 'block' : 'none';
    $('btnCc').classList.toggle('cc-active', isCcEnabled);
  }

  function toggleFullscreen() {
    const container = $('ytPlayerContainer');
    if (!document.fullscreenElement) {
      container.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }

  // --------------------------------------------------------------------------
  // Playback Engine & Crossfade
  // --------------------------------------------------------------------------
  function remainingBuffer() {
    if (!state?.segments?.length) return 0;
    let total = 0;
    for (let i = Math.max(0, currentIndex); i < state.segments.length; i++) {
      total += Number(state.segments[i].duration || 0);
    }
    if (currentIndex >= 0) {
      total -= players[front].currentTime || 0;
    }
    return Math.max(0, total);
  }

  function updateBufferProgress() {
    const player = players[front];
    const dur = player.duration || 1;
    const current = player.currentTime || 0;
    const pct = Math.min(100, (current / dur) * 100);

    $('scrubberPlayed').style.width = `${pct}%`;
    $('currentTimeDisplay').textContent = `${fmt(current)} / ${fmt(dur)}`;
  }

  function showScene(index) {
    const item = state?.segments?.[index];
    if (!item) return;
    currentIndex = index;

    $('primaryVideoTitle').textContent = item.translated_title ? `${item.title} · ${item.translated_title}` : item.title;
    $('sceneNoBadge').textContent = `Scene ${item.number}`;

    // Render in YouTube Closed Captions
    if (subtitleManager) {
      subtitleManager.render(item);
    }

    // Update Description Box
    $('descViews').textContent = `${(state.segments || []).length} scenes`;
    $('descDate').textContent = item.motion_repeated ? 'Forward/backward coverage' : 'Continuous slow motion';
    $('descPromptText').textContent = state.config?.prompt || '';

    const lp = (item.learning_point || '').trim();
    const source = item.sources?.[0];
    const factBox = $('descFactBox');
    if (factBox) {
      factBox.hidden = !lp;
      factBox.innerHTML = lp
        ? `<b>Verified offline fact:</b> ${esc(lp)}${
            source ? ` <a href="${esc(source.url)}" target="_blank" rel="noreferrer">${esc(source.title)}</a>` : ''
          }`
        : '';
    }

    renderSidebar();
  }

  function preloadNext() {
    const next = currentIndex + 1;
    if (!state?.segments?.[next]) return;
    const back = front === 'A' ? 'B' : 'A';
    if (players[back].dataset.number !== String(state.segments[next].number)) {
      players[back].src = mediaUrl(state.segments[next].path);
      players[back].dataset.number = state.segments[next].number;
      players[back].load();
    }
  }

  async function playIndex(index) {
    if (!state?.segments?.[index]) return false;
    const item = state.segments[index];
    const nextFront = front === 'A' ? 'B' : 'A';
    const candidate = currentIndex < 0 ? front : nextFront;
    const player = players[candidate];

    if (player.dataset.number !== String(item.number)) {
      player.src = mediaUrl(item.path);
      player.dataset.number = item.number;
      player.load();
    }
    player.muted = userMuted;

    try {
      await player.play();
    } catch {
      $('btnPlayPause').innerHTML = `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
      return false;
    }

    players[front].classList.remove('live');
    front = candidate;
    player.classList.add('live');
    $('playerWaiting').style.display = 'none';
    playbackStarted = true;
    $('btnPlayPause').innerHTML = `<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
    showScene(index);
    preloadNext();
    return true;
  }

  async function advance() {
    const next = currentIndex + 1;
    if (state?.segments?.[next]) {
      await playIndex(next);
      return;
    }
    $('playerWaiting').style.display = 'grid';
    $('waitTitle').textContent = 'Generating Next Scene';
    $('waitText').textContent = 'Continuity buffer is preparing the next unique visual asset.';
    playbackStarted = false;
  }

  players.A.addEventListener('ended', advance);
  players.B.addEventListener('ended', advance);
  players.A.addEventListener('timeupdate', () => {
    if (front === 'A') updateBufferProgress();
  });
  players.B.addEventListener('timeupdate', () => {
    if (front === 'B') updateBufferProgress();
  });

  // --------------------------------------------------------------------------
  // YouTube Sidebar: Up Next Recommendations & Live Chat
  // --------------------------------------------------------------------------
  function renderSidebar() {
    if (activeSidebarTab === 'scenes') {
      renderSceneRecommendations();
    } else if (activeSidebarTab === 'chat') {
      renderLiveChat();
    } else if (activeSidebarTab === 'saved') {
      renderSavedSessions();
    }
  }

  function renderSceneRecommendations() {
    const body = $('sidebarBody');
    body.innerHTML = '';

    const segments = state?.segments || [];
    if (!segments.length) {
      body.innerHTML = `<div style="color:var(--yt-text-secondary);padding:16px;text-align:center;">Planning stream scenes…</div>`;
      return;
    }

    segments.forEach((s, idx) => {
      const card = document.createElement('div');
      card.className = `yt-video-card ${idx === currentIndex ? 'active' : ''}`;
      card.innerHTML = `
        <div class="yt-thumb-box">
          <span class="yt-thumb-scene-num">#${s.number}</span>
          <span class="yt-thumb-duration">${fmt(s.duration)}</span>
        </div>
        <div class="yt-video-meta">
          <div class="yt-video-title">${esc(s.title)}</div>
          <div class="yt-video-channel">Scene ${s.number} · Wan 2.2</div>
          <div class="yt-video-stats">${s.motion_repeated ? 'Coverage motion' : 'Slow motion'}</div>
        </div>
      `;
      card.onclick = () => playIndex(idx);
      body.appendChild(card);
    });
  }

  let chatScope = 'next_scene'; // 'audience_message', 'next_scene', 'persistent'
  let chatDelivery = 'after_buffer'; // 'after_buffer', 'next_unrendered'

  function getConsequenceHint() {
    if (chatScope === 'audience_message') {
      return 'Host acknowledges and answers your message naturally in an upcoming turn.';
    }
    if (chatScope === 'next_scene') {
      return chatDelivery === 'after_buffer'
        ? 'Non-disruptive one-scene event queued after buffer. Existing work continues.'
        : 'Fast steering: cancels speculative planning and rebuilds next unrendered scene.';
    }
    if (chatScope === 'persistent') {
      return chatDelivery === 'after_buffer'
        ? 'Lasting world rule queued safely after buffer. Applies to all future scenes.'
        : 'Lasting world rule applied immediately on next unrendered scene.';
    }
    return '';
  }

  function renderLiveChat() {
    const body = $('sidebarBody');
    const isInteractive = state?.config?.mode === 'interactive';
    if (isInteractive && chatScope !== 'audience_message' && chatScope !== 'persistent' && chatScope !== 'next_scene') {
      chatScope = 'audience_message';
    } else if (!isInteractive && chatScope === 'audience_message') {
      chatScope = 'next_scene';
    }

    body.innerHTML = `
      <div class="yt-chat-container">
        <div class="yt-chat-banner" id="chatBanner">
          ${isInteractive ? 'Interactive Mode: Chat with host character or direct world.' : 'Live Directives: Steer living world events and lasting rules.'}
        </div>
        <div class="yt-chat-messages" id="chatMsgList"></div>
        <div class="yt-chat-input-row">
          <div class="yt-chat-options-bar">
            <div class="yt-scope-group">
              ${isInteractive ? `<button class="yt-mini-chip ${chatScope === 'audience_message' ? 'active' : ''}" type="button" id="chipScopeChat">💬 Chat Host</button>` : ''}
              <button class="yt-mini-chip ${chatScope === 'next_scene' ? 'active' : ''}" type="button" id="chipScopeEvent">⚡ Event</button>
              <button class="yt-mini-chip ${chatScope === 'persistent' ? 'active' : ''}" type="button" id="chipScopeRule">📜 Rule</button>
            </div>
            <div class="yt-delivery-group">
              <button class="yt-mini-chip ${chatDelivery === 'after_buffer' ? 'active' : ''}" type="button" id="chipDeliveryBuffer" title="Zero disruption — existing buffer continues">⏱️ Buffer</button>
              <button class="yt-mini-chip ${chatDelivery === 'next_unrendered' ? 'active' : ''}" type="button" id="chipDeliveryFast" title="Fast replan on next unrendered scene">⚡ Fast</button>
            </div>
          </div>
          <div class="yt-consequence-hint" id="chatConsequenceHint">${getConsequenceHint()}</div>
          <div class="yt-chat-input-box">
            <textarea class="yt-chat-input" id="chatInput" placeholder="${
              chatScope === 'audience_message' ? 'Ask host a question or speak…' : chatScope === 'persistent' ? 'Add lasting world rule…' : 'Direct next scene event…'
            }"></textarea>
            <button class="yt-chat-send-btn" id="btnSendChat" type="button">Send</button>
          </div>
        </div>
      </div>
    `;

    const list = $('chatMsgList');
    const items = (state?.live_directives || []).filter((i) => ['pending', 'active'].includes(i.status));

    if (!items.length) {
      list.innerHTML = `<div style="color:var(--yt-text-muted);font-size:12px;text-align:center;padding:20px 0;">No active directives yet.<br>Choose a type above and send to steer the living story.</div>`;
    } else {
      items.forEach((item) => {
        const msg = document.createElement('div');
        msg.className = 'yt-chat-msg';
        const isRule = item.scope === 'persistent';
        const isEvent = item.scope === 'next_scene';
        const isChat = item.scope === 'audience_message';
        const scopeBadge = isChat ? 'Chat' : isRule ? 'Rule' : 'Event';
        const scopeClass = isRule ? 'rule' : isEvent ? 'event' : '';
        const timingLabel = item.delivery === 'next_unrendered' ? 'Fast' : `Scene #${item.activation_scene || '?'}`;

        msg.innerHTML = `
          <div class="yt-chat-avatar">${isChat ? '💬' : isRule ? '📜' : '⚡'}</div>
          <div class="yt-chat-content">
            <div class="yt-chat-header-row">
              <span class="yt-chat-badge ${scopeClass}">${scopeBadge}</span>
              <span class="yt-chat-timing-badge">${timingLabel}</span>
              <span class="yt-chat-author">Viewer</span>
              ${isRule && item.status === 'active' ? `<button class="yt-chat-remove-btn" data-id="${item.id}" type="button">✕ Remove</button>` : ''}
            </div>
            <div class="yt-chat-text">${esc(item.text)}</div>
          </div>
        `;
        list.appendChild(msg);
      });
    }

    // Attach Remove Listeners
    list.querySelectorAll('.yt-chat-remove-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeDirective(btn.dataset.id);
      });
    });

    // Scope chip events
    $('chipScopeChat')?.addEventListener('click', () => {
      chatScope = 'audience_message';
      renderLiveChat();
    });
    $('chipScopeEvent')?.addEventListener('click', () => {
      chatScope = 'next_scene';
      renderLiveChat();
    });
    $('chipScopeRule')?.addEventListener('click', () => {
      chatScope = 'persistent';
      renderLiveChat();
    });

    // Delivery chip events
    $('chipDeliveryBuffer')?.addEventListener('click', () => {
      chatDelivery = 'after_buffer';
      renderLiveChat();
    });
    $('chipDeliveryFast')?.addEventListener('click', () => {
      chatDelivery = 'next_unrendered';
      renderLiveChat();
    });

    $('btnSendChat')?.addEventListener('click', sendDirectFromChat);
    $('chatInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendDirectFromChat();
      }
    });
  }

  async function removeDirective(directiveId) {
    if (!activeId || !directiveId) return;
    try {
      const r = await fetch(`/api/theater/${activeId}/directives/${directiveId}`, {
        method: 'DELETE',
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not remove directive');
      renderState(data);
      toast('World rule removed');
    } catch (e) {
      toast(e.message);
    }
  }

  async function sendDirectFromChat() {
    if (!activeId || directiveRequestInFlight) return;
    const input = $('chatInput');
    const text = (input?.value || '').trim();
    if (!text) return;

    directiveRequestInFlight = true;
    try {
      const r = await fetch(`/api/theater/${activeId}/directives`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, scope: chatScope, delivery: chatDelivery }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not send directive');
      input.value = '';
      renderState(data);
      const toastMsg = chatDelivery === 'after_buffer'
        ? `Queued from scene ${data.live_directives?.slice(-1)[0]?.activation_scene || 'ahead'}`
        : 'Fast steering triggered on next unrendered scene';
      toast(toastMsg);
    } catch (e) {
      toast(e.message);
    } finally {
      directiveRequestInFlight = false;
    }
  }

  async function renderSavedSessions() {
    const body = $('sidebarBody');
    body.innerHTML = '<div style="color:var(--yt-text-secondary);padding:16px;text-align:center;">Loading saved streams…</div>';

    try {
      const { sessions } = await (await fetch('/api/theater')).json();
      body.innerHTML = '';
      if (!sessions || !sessions.length) {
        body.innerHTML = '<div style="color:var(--yt-text-secondary);padding:16px;text-align:center;">No saved streams yet.</div>';
        return;
      }
      sessions.slice(0, 15).forEach((s) => {
        const card = document.createElement('div');
        card.className = 'yt-video-card';
        card.innerHTML = `
          <div class="yt-thumb-box">
            <span class="yt-thumb-scene-num">▶</span>
            <span class="yt-thumb-duration">${fmt(s.total_duration)}</span>
          </div>
          <div class="yt-video-meta">
            <div class="yt-video-title">${esc(s.title || s.config.prompt)}</div>
            <div class="yt-video-channel">${(s.segments || []).length} scenes · ${s.status}</div>
            <div class="yt-video-stats">Saved Stream</div>
          </div>
        `;
        card.onclick = () => poll(s.id);
        body.appendChild(card);
      });
    } catch {
      body.innerHTML = '<div style="color:var(--yt-text-secondary);padding:16px;text-align:center;">Could not load saved streams.</div>';
    }
  }

  // --------------------------------------------------------------------------
  // State Polling & Updates
  // --------------------------------------------------------------------------
  function renderState(next) {
    state = next;
    activeId = next.id;
    localStorage.setItem('wanTheaterSession', activeId);

    const isRunning = ['starting', 'planning', 'generating', 'narrating', 'buffering', 'running'].includes(next.status);
    $('btnStopStream').disabled = !isRunning;
    $('liveStatusBadge').textContent = isRunning ? 'LIVE' : next.status.toUpperCase();

    $('primaryVideoTitle').textContent = next.title || next.config?.prompt || 'Endless Story';

    renderSidebar();

    if (next.status === 'failed' || next.status === 'interrupted' || next.status === 'stopped') {
      $('playerWaiting').style.display = 'grid';
      $('waitTitle').textContent = next.status === 'failed' ? 'Stream Paused' : 'Stream Stopped';
      $('waitText').textContent = next.message;
      clearTimeout(pollTimer);
      return;
    }

    if (!playbackStarted && next.segments?.length >= 2) {
      playIndex(Math.max(0, currentIndex));
    } else if (playbackStarted && next.segments?.length > lastSegmentCount) {
      preloadNext();
    }
    lastSegmentCount = next.segments?.length || 0;

    clearTimeout(pollTimer);
    pollTimer = setTimeout(() => poll(activeId), 1500);
  }

  async function poll(id) {
    try {
      const r = await fetch(`/api/theater/${id}`);
      if (!r.ok) throw new Error();
      renderState(await r.json());
    } catch {
      pollTimer = setTimeout(() => poll(id), 2500);
    }
  }

  async function stopStream() {
    if (!activeId) return;
    $('btnStopStream').disabled = true;
    try {
      const r = await fetch(`/api/theater/${activeId}/stop`, { method: 'POST' });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not stop stream');
      players.A.pause();
      players.B.pause();
      renderState(data);
      toast('Stream stopped. Complete archive preserved.');
    } catch (e) {
      $('btnStopStream').disabled = false;
      toast(e.message);
    }
  }

  async function loadRecent() {
    if (activeSidebarTab === 'saved') {
      renderSavedSessions();
    }
  }

  // --------------------------------------------------------------------------
  // Settings Modal Flyout
  // --------------------------------------------------------------------------
  function openSettingsModal() {
    $('ytSettingsModal')?.classList.add('open');
  }

  function closeSettingsModal() {
    $('ytSettingsModal')?.classList.remove('open');
  }

  async function previewVoiceSample() {
    const btn = $('btnPreviewVoice');
    const audio = $('voiceSampleAudio');
    btn.disabled = true;
    btn.textContent = 'Rendering…';

    try {
      const r = await fetch('/api/theater/voice-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          voice: $('voiceSelect')?.value || 'M1',
          language: $('languageSelect')?.value || 'en',
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Voice preview failed');
      audio.src = `${mediaUrl(data.path)}&v=${Date.now()}`;
      audio.hidden = false;
      await audio.play();
    } catch (e) {
      toast(e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Hear Voice';
    }
  }

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    // Draggable CC Subtitles
    const captionOverlay = $('ytCaptionWindow');
    const playerContainer = $('ytPlayerContainer');
    if (captionOverlay && playerContainer && window.SubtitleManager) {
      subtitleManager = new window.SubtitleManager(captionOverlay, playerContainer);
      $('btnResetCaptionPos')?.addEventListener('click', () => subtitleManager.resetPosition());
    }

    // Top Search Story Submission
    $('ytSearchForm')?.addEventListener('submit', submitStoryPrompt);

    // Player Controls
    $('btnPlayPause')?.addEventListener('click', togglePlayPause);
    $('btnNextScene')?.addEventListener('click', () => {
      if (state?.segments?.[currentIndex + 1]) playIndex(currentIndex + 1);
    });
    $('btnMute')?.addEventListener('click', toggleMute);
    $('btnCc')?.addEventListener('click', toggleCc);
    $('btnFullscreen')?.addEventListener('click', toggleFullscreen);

    // Action Row
    $('btnStopStream')?.addEventListener('click', stopStream);
    $('btnOpenSettings')?.addEventListener('click', openSettingsModal);
    $('btnCloseSettings')?.addEventListener('click', closeSettingsModal);
    $('btnPreviewVoice')?.addEventListener('click', previewVoiceSample);

    // Sidebar Tab Switching
    $('tabScenes')?.addEventListener('click', () => {
      activeSidebarTab = 'scenes';
      $('tabScenes').classList.add('active');
      $('tabChat').classList.remove('active');
      $('tabSaved').classList.remove('active');
      renderSidebar();
    });

    $('tabChat')?.addEventListener('click', () => {
      activeSidebarTab = 'chat';
      $('tabChat').classList.add('active');
      $('tabScenes').classList.remove('active');
      $('tabSaved').classList.remove('active');
      renderSidebar();
    });

    $('tabSaved')?.addEventListener('click', () => {
      activeSidebarTab = 'saved';
      $('tabSaved').classList.add('active');
      $('tabScenes').classList.remove('active');
      $('tabChat').classList.remove('active');
      renderSidebar();
    });

    // Populate Translation Options
    const langSelect = $('languageSelect');
    const transSelect = $('transLanguageSelect');
    if (langSelect && transSelect) {
      [...langSelect.options].forEach((opt) => {
        if (opt.value && opt.value !== 'na') {
          const c = document.createElement('option');
          c.value = opt.value;
          c.textContent = opt.textContent;
          transSelect.appendChild(c);
        }
      });
    }

    const remembered = localStorage.getItem('wanTheaterSession');
    if (remembered) poll(remembered);
  });
})();
