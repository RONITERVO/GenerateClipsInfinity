/**
 * ============================================================================
 * ENDLESS OFFLINE THEATER CONTROLLER (theater.js)
 * ============================================================================
 * Manages playback, state polling, live steering directives, and telemetry.
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
  let lastSegmentCount = 0;
  let directiveRequestInFlight = false;
  let subtitleManager = null;

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
    return m ? `${m}m ${s}s` : `${s}s`;
  }

  function mediaUrl(path) {
    return `/api/video?path=${encodeURIComponent(path)}`;
  }

  function populateTranslationLanguages() {
    const target = $('translationLanguage');
    const source = $('language');
    if (!target || !source) return;

    [...source.options]
      .filter((opt) => opt.value && opt.value !== 'na')
      .forEach((opt) => {
        const copy = document.createElement('option');
        copy.value = opt.value;
        copy.textContent = opt.textContent;
        target.appendChild(copy);
      });
    syncTranslationLanguages();
  }

  function syncTranslationLanguages() {
    const source = $('language').value;
    const target = $('translationLanguage');
    if (!target) return;

    [...target.options].forEach((option) => {
      option.disabled = Boolean(option.value && option.value === source);
    });
    if (target.value === source) target.value = '';

    const active = target.options[target.selectedIndex];
    const note = $('translationNote');
    if (note) {
      note.textContent = target.value
        ? `Each ${$('language').options[$('language').selectedIndex].textContent} sentence will be spoken and shown, followed by ${active.textContent}. The total speech duration stays aligned.`
        : 'Optional: after every story sentence, show and speak its translation.';
    }
  }

  function syncLiveExperience(mode) {
    const interactive = mode === 'interactive';
    const dream = mode === 'dream';
    const chatOption = [...$('liveScope').options].find((opt) => opt.value === 'audience_message');

    if (chatOption) {
      const wasDisabled = chatOption.disabled;
      chatOption.hidden = !interactive;
      chatOption.disabled = !interactive;
      if (interactive && wasDisabled) $('liveScope').value = 'audience_message';
      if (!interactive && $('liveScope').value === 'audience_message') $('liveScope').value = 'next_scene';
    }

    $('liveControlTitle').textContent = interactive
      ? 'Chat with the character or decide what happens'
      : dream
      ? 'Whisper into the dream'
      : 'Direct the world while it runs';

    $('liveControlHelp').textContent = interactive
      ? 'Type to the recurring host, decide a later moment, or set a persistent show rule. Delayed chat preserves every prepared scene. The host answers aloud when its reserved scene arrives.'
      : dream
      ? 'Add another image or feeling. It enters as a later association without cancelling prepared scenes or turning the dream into a factual explanation.'
      : 'By default direction waits behind every scene Gemma already planned, wasting no work and creating no gap. Fast steering replaces speculative text, but never changes completed media.';

    $('liveText').placeholder = interactive
      ? 'For example: What surprised you most today?'
      : dream
      ? 'For example: warm rain'
      : 'For example: A sudden summer storm forces everyone into the old lighthouse.';

    $('liveSend').textContent = interactive ? 'Send to character' : dream ? 'Whisper' : 'Direct story';

    if (!activeId) {
      $('liveEffect').textContent = dream
        ? 'Start a dream to add delayed associations. Whispers never add encyclopedia grounding.'
        : 'Start or open a theater session to direct it. Directions cannot override premise or verified facts.';
    }
  }

  function syncExperienceSetup() {
    const mode = $('mode').value;
    const interactive = mode === 'interactive';
    const dream = mode === 'dream';

    $('promptLabel').textContent = interactive
      ? 'Character, setting, and ongoing activity'
      : dream
      ? 'Dream seed · one word is enough'
      : 'Story, world, or topic';

    $('prompt').placeholder = interactive
      ? 'A friendly night-shift lighthouse keeper restores old instruments, tells stories about the coast, and lets delayed viewer messages shape the evening.'
      : dream
      ? 'Velvet'
      : 'A curious class explores the history of astronomy by traveling through a magical observatory where every discovery changes the night sky.';

    $('experienceIntro').textContent = interactive
      ? 'Create a resident on-screen character who keeps living their story and answers delayed text chat in future synchronized scenes.'
      : dream
      ? 'Offer a faint pre-sleep cue. Gemma invents the people, places, rules, and unfolding imagery without encyclopedia grounding or factual explanation.'
      : 'One prompt begins a continuous local show. Gemma 4 E4B writes on CPU while the RTX 5070 creates each new scene.';

    $('learningField').hidden = dream;
    $('learning').disabled = dream;
    if (dream) $('learning').value = '';
    if (!activeId) syncLiveExperience(mode);
  }

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

  function updateBuffer() {
    const seconds = remainingBuffer();
    const pct = Math.min(100, (seconds / 180) * 100);
    $('bufferTime').textContent = fmt(seconds);
    $('bufferBar').style.width = `${pct}%`;
    $('bufferHealth').textContent =
      seconds > 120 ? 'Protected' : seconds > 45 ? 'Building safely' : seconds > 0 ? 'Low — production adapting' : 'Waiting for unique scenes';
  }

  function showScene(index) {
    const item = state.segments[index];
    if (!item) return;
    currentIndex = index;

    $('sceneNo').textContent = `Scene ${item.number} · ${
      item.motion_repeated ? 'protected forward/backward coverage' : 'continuous slow motion'
    }`;
    $('sceneTitle').textContent = item.translated_title ? `${item.title} · ${item.translated_title}` : item.title;

    // Render in real-time draggable subtitle overlay
    if (subtitleManager) {
      subtitleManager.render(item);
    }

    const lp = (item.learning_point || '').trim();
    const source = item.sources?.[0];
    const learningEl = $('learningPoint');
    learningEl.hidden = !lp;
    learningEl.innerHTML = lp
      ? `<b>Verified offline fact:</b> ${esc(lp)}${
          source ? ` <a href="${esc(source.url)}" target="_blank" rel="noreferrer">${esc(source.title)}</a>` : ''
        }`
      : '';

    renderQueue();
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
      $('resumePlay').style.display = 'inline-flex';
      $('waiting').style.display = 'grid';
      $('waitTitle').textContent = 'Your browser paused the theater';
      $('waitText').textContent = 'The first visual and narration are synchronized and ready.';
      return false;
    }

    players[front].classList.remove('live');
    front = candidate;
    player.classList.add('live');
    $('waiting').style.display = 'none';
    playbackStarted = true;
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
    $('waiting').style.display = 'grid';
    $('waitTitle').textContent = 'Protecting story continuity';
    $('waitText').textContent =
      'The next unique scene is still being made. Playback will resume automatically without reusing an earlier asset.';
    playbackStarted = false;
  }

  players.A.addEventListener('ended', advance);
  players.B.addEventListener('ended', advance);

  players.A.addEventListener('timeupdate', () => {
    if (front === 'A') {
      updateBuffer();
      $('clock').textContent = fmt(players.A.currentTime);
    }
  });

  players.B.addEventListener('timeupdate', () => {
    if (front === 'B') {
      updateBuffer();
      $('clock').textContent = fmt(players.B.currentTime);
    }
  });

  function renderQueue() {
    const root = $('queue');
    root.innerHTML = '';
    if (!state?.segments?.length) {
      root.innerHTML = '<span class="sub">Planning and generation are underway.</span>';
      return;
    }
    state.segments.forEach((s, i) => {
      const el = document.createElement('div');
      el.className = `scene-chip ${i === currentIndex ? 'current' : ''}`;
      el.innerHTML = `<b>${s.number}. ${esc(s.title)}</b><small>${fmt(s.duration)} · ${
        s.motion_repeated ? 'motion coverage' : 'slow motion'
      }</small>`;
      root.appendChild(el);
    });
  }

  function renderDirectives() {
    const root = $('directiveList');
    const items = (state?.live_directives || []).filter((item) => ['pending', 'active'].includes(item.status));
    root.innerHTML = '';

    $('liveText').disabled = !activeId;
    $('liveScope').disabled = !activeId;
    $('liveDelivery').disabled = !activeId;
    $('liveSend').disabled = !activeId;

    if (!items.length) {
      root.innerHTML = '<span class="sub">No pending chat, events, or persistent rules.</span>';
    }

    items.forEach((item) => {
      const el = document.createElement('div');
      el.className = 'directive';
      const kind = document.createElement('em');
      kind.textContent = item.scope === 'audience_message' ? 'Chat' : item.scope === 'persistent' ? 'Persistent' : 'One scene';
      const timing = document.createElement('em');
      timing.textContent = item.first_applied_scene
        ? `since scene ${item.first_applied_scene}`
        : item.delivery === 'after_buffer'
        ? `from scene ${item.activation_scene}`
        : 'fast';
      const words = document.createElement('span');
      words.textContent = item.text;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '✕';
      remove.title = 'Remove from future scenes';
      remove.onclick = () => removeDirective(item.id);
      el.append(kind, timing, words, remove);
      root.appendChild(el);
    });

    const scheduled = items
      .filter((item) => item.delivery === 'after_buffer' && !item.first_applied_scene)
      .map((item) => Number(item.activation_scene || 0))
      .filter(Boolean);
    const delayed = state?.metrics?.live_steering_legacy_delay_through_scene;
    const revised = state?.metrics?.last_live_steering_scene;

    $('liveEffect').textContent = scheduled.length
      ? `Non-disruptive input queued from scene ${Math.min(
          ...scheduled
        )}. Existing plans, speech preparation, and rendering continue unchanged.`
      : delayed
      ? `This older saved buffer has no rollback checkpoint. The input will take effect after scene ${delayed}.`
      : revised
      ? `Fast steering rebuilt speculative planning from scene ${revised}. Visual rendering was kept intact.`
      : 'Default input waits behind the planned buffer and does not cancel work.';
  }

  async function submitDirective() {
    if (!activeId || directiveRequestInFlight) return;
    const text = $('liveText').value.trim();
    if (!text) {
      toast('Enter a message or direction');
      return;
    }
    const button = $('liveSend');
    const delivery = $('liveDelivery').value;
    const scope = $('liveScope').value;

    directiveRequestInFlight = true;
    button.disabled = true;

    try {
      const r = await fetch(`/api/theater/${activeId}/directives`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, scope, delivery }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not send input');
      $('liveText').value = '';
      renderState(data);
      toast(
        delivery === 'after_buffer'
          ? scope === 'audience_message'
            ? 'Chat queued without replacing planned work'
            : 'Direction queued without replacing planned work'
          : 'Fast input queued for unrendered scene'
      );
    } catch (e) {
      toast(e.message);
    } finally {
      directiveRequestInFlight = false;
      button.disabled = !activeId;
    }
  }

  async function removeDirective(id) {
    if (!activeId) return;
    try {
      const r = await fetch(`/api/theater/${activeId}/directives/${encodeURIComponent(id)}`, { method: 'DELETE' });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not remove the rule');
      renderState(data);
      toast('Removed from future unrendered scenes');
    } catch (e) {
      toast(e.message);
    }
  }

  function renderState(next) {
    state = next;
    activeId = next.id;
    syncLiveExperience(next.config?.mode);
    localStorage.setItem('wanTheaterSession', activeId);

    $('stop').disabled = !['starting', 'planning', 'generating', 'narrating', 'buffering', 'running'].includes(next.status);
    $('continueStory').hidden = !['interrupted', 'stopped', 'failed'].includes(next.status);
    $('start').disabled = !$('stop').disabled;
    $('deskTitle').textContent = next.title || 'Preparing endless story';

    const writerMode =
      next.metrics?.writer_mode === 'gpu_burst'
        ? 'RTX text-buffer burst'
        : next.metrics?.writer_mode === 'cpu_sustain'
        ? 'CPU writer + RTX visual worker'
        : 'local models starting';
    $('machine').textContent = `${next.status} · ${writerMode}`;

    $('readyCount').textContent = (next.segments || []).length;
    $('plannerSpeed').textContent = next.metrics?.planner_tps ? `${next.metrics.planner_tps} tok/s` : 'loading';
    $('productionTime').textContent = next.metrics?.production_ema ? fmt(next.metrics.production_ema) : 'measuring';
    $('coverage').textContent = next.metrics?.coverage_ratio ? `${next.metrics.coverage_ratio}×` : 'measuring';

    $('waitTitle').textContent = next.segments?.length < 2 ? 'Building the safe opening buffer' : next.message;
    $('waitText').textContent =
      next.segments?.length < 2
        ? `${next.message} Narration begins after two complete synchronized scenes.`
        : next.message;

    renderQueue();
    updateBuffer();
    renderDirectives();

    if (next.status === 'failed' || next.status === 'interrupted' || next.status === 'stopped') {
      $('waiting').style.display = 'grid';
      $('waitTitle').textContent = next.status === 'failed' ? 'The theater needs attention' : 'Saved theater paused';
      $('waitText').textContent = next.message;
      $('resumePlay').style.display = next.segments?.length ? 'inline-flex' : 'none';
      $('start').disabled = false;
      clearTimeout(pollTimer);
      loadRecent();
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

  async function start(event) {
    event.preventDefault();
    const quality_settings = {
      width: Number($('qualityWidth').value),
      height: Number($('qualityHeight').value),
      frames: Number($('qualityFrames').value),
      fps: Number($('qualityFps').value),
      min_words: Number($('minWords').value),
      max_words: Number($('maxWords').value),
      max_slow: Number($('maxSlow').value),
    };
    const payload = {
      prompt: $('prompt').value,
      learning_focus: $('learning').value,
      mode: $('mode').value,
      audience: $('audience').value,
      language: $('language').value,
      translation_language: $('translationLanguage').value,
      voice: $('voice').value,
      quality_settings,
      context_compaction_scenes: Number($('contextCompactionScenes').value),
      seed: Number($('seed').value),
    };

    $('start').disabled = true;
    $('waiting').style.display = 'grid';
    $('waitTitle').textContent = 'Starting the local story engine';
    $('waitText').textContent =
      'Gemma first builds a short story buffer and releases the RTX GPU for Wan. No narration plays until its visual is ready.';

    try {
      const r = await fetch('/api/theater', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not start theater.');

      currentIndex = -1;
      playbackStarted = false;
      lastSegmentCount = 0;
      renderState(data);
      loadRecent();
    } catch (e) {
      $('start').disabled = false;
      toast(e.message);
    }
  }

  async function previewVoice() {
    const button = $('previewVoice');
    const sample = $('voiceSample');
    button.disabled = true;
    button.textContent = 'Rendering…';

    try {
      const r = await fetch('/api/theater/voice-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: $('voice').value, language: $('language').value }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Voice preview failed');
      sample.src = `${mediaUrl(data.path)}&v=${Date.now()}`;
      sample.hidden = false;
      await sample.play();
    } catch (e) {
      toast(e.message);
    } finally {
      button.disabled = false;
      button.textContent = 'Hear voice';
    }
  }

  async function stop() {
    if (!activeId) return;
    $('stop').disabled = true;
    try {
      const r = await fetch(`/api/theater/${activeId}/stop`, { method: 'POST' });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Could not stop');
      players.A.pause();
      players.B.pause();
      renderState(data);
      toast('The complete stream archive was kept');
    } catch (e) {
      $('stop').disabled = false;
      toast(e.message || 'Could not stop and archive');
    }
  }

  async function resumeSession(id) {
    const r = await fetch(`/api/theater/${id}/resume`, { method: 'POST' });
    const data = await r.json();
    if (r.ok) {
      currentIndex = -1;
      playbackStarted = false;
      renderState(data);
    } else {
      toast(data.error || 'Could not continue');
    }
  }

  async function loadRecent() {
    try {
      const { sessions } = await (await fetch('/api/theater')).json();
      const root = $('recent');
      root.innerHTML = '';
      if (!sessions || !sessions.length) {
        root.innerHTML = '<span class="sub">No sessions yet.</span>';
        return;
      }
      sessions.slice(0, 8).forEach((s) => {
        const el = document.createElement('div');
        el.className = 'scene-chip';
        el.style.display = 'flex';
        el.style.justifyContent = 'space-between';
        el.style.alignItems = 'center';
        el.style.cursor = 'pointer';
        el.innerHTML = `<div><b>${esc(s.title || s.config.prompt)}</b><small>${s.status} · ${(s.segments || []).length} scenes · ${fmt(
          s.total_duration
        )}</small></div><button class="btn-ghost" style="padding:4px 8px;font-size:10px;">Open</button>`;
        el.onclick = () => poll(s.id);
        root.appendChild(el);
      });
    } catch {}
  }

  // Initialize UI on Load
  document.addEventListener('DOMContentLoaded', () => {
    // Initialize YouTube-inspired Draggable Subtitles
    const captionOverlay = $('captionOverlay');
    const screenViewport = $('screen');
    if (captionOverlay && screenViewport && window.SubtitleManager) {
      subtitleManager = new window.SubtitleManager(captionOverlay, screenViewport);
      $('captionResetPos')?.addEventListener('click', () => subtitleManager.resetPosition());
    }

    $('setup').addEventListener('submit', start);
    $('mode').addEventListener('change', syncExperienceSetup);
    $('language').addEventListener('change', syncTranslationLanguages);
    $('translationLanguage').addEventListener('change', syncTranslationLanguages);
    $('previewVoice').addEventListener('click', previewVoice);
    $('liveSend').addEventListener('click', submitDirective);
    $('liveText').addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') submitDirective();
    });
    $('stop').addEventListener('click', stop);
    $('continueStory').addEventListener('click', () => resumeSession(activeId));
    $('resumePlay').addEventListener('click', () => playIndex(Math.max(0, currentIndex)));
    $('mute').addEventListener('click', () => {
      userMuted = !userMuted;
      players.A.muted = userMuted;
      players.B.muted = userMuted;
      $('mute').textContent = userMuted ? 'Sound off' : 'Sound on';
    });
    $('fullscreen').addEventListener('click', () => $('screen').requestFullscreen?.());

    populateTranslationLanguages();
    syncExperienceSetup();
    loadRecent();

    const remembered = localStorage.getItem('wanTheaterSession');
    if (remembered) poll(remembered);
  });
})();
