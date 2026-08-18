/**
 * ============================================================================
 * YOUTUBE-INSPIRED REAL-TIME DRAGGABLE SUBTITLES (subtitles.js)
 * ============================================================================
 * Provides draggable caption overlay mechanics, boundary clamping,
 * paired bilingual sentence rendering, and position persistence.
 * ============================================================================
 */

class SubtitleManager {
  constructor(overlayEl, containerEl) {
    this.overlay = overlayEl;
    this.container = containerEl;
    this.isDragging = false;
    this.startX = 0;
    this.startY = 0;
    this.initialLeft = 0;
    this.initialTop = 0;
    this.currentPosition = null;

    this._initDraggable();
    this._restorePosition();
  }

  _initDraggable() {
    if (!this.overlay || !this.container) return;

    const onPointerDown = (e) => {
      // Don't drag if clicking tools or buttons inside caption
      if (e.target.closest('.caption-tool-btn')) return;

      this.isDragging = true;
      this.overlay.classList.add('dragging');

      const rect = this.overlay.getBoundingClientRect();
      const containerRect = this.container.getBoundingClientRect();

      this.startX = e.clientX;
      this.startY = e.clientY;
      this.initialLeft = rect.left - containerRect.left;
      this.initialTop = rect.top - containerRect.top;

      // Remove center-transform once user begins dragging
      this.overlay.style.transform = 'none';
      this.overlay.style.bottom = 'auto';
      this.overlay.style.left = `${this.initialLeft}px`;
      this.overlay.style.top = `${this.initialTop}px`;

      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
    };

    const onPointerMove = (e) => {
      if (!this.isDragging) return;

      const deltaX = e.clientX - this.startX;
      const deltaY = e.clientY - this.startY;

      const containerRect = this.container.getBoundingClientRect();
      const overlayRect = this.overlay.getBoundingClientRect();

      let newLeft = this.initialLeft + deltaX;
      let newTop = this.initialTop + deltaY;

      // Clamp within video stage viewport
      const minLeft = 12;
      const maxLeft = containerRect.width - overlayRect.width - 12;
      const minTop = 12;
      const maxTop = containerRect.height - overlayRect.height - 12;

      newLeft = Math.max(minLeft, Math.min(newLeft, maxLeft));
      newTop = Math.max(minTop, Math.min(newTop, maxTop));

      this.overlay.style.left = `${newLeft}px`;
      this.overlay.style.top = `${newTop}px`;

      this.currentPosition = {
        leftPct: newLeft / containerRect.width,
        topPct: newTop / containerRect.height,
      };
    };

    const onPointerUp = () => {
      if (!this.isDragging) return;
      this.isDragging = false;
      this.overlay.classList.remove('dragging');

      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);

      if (this.currentPosition) {
        localStorage.setItem('wan_theater_caption_pos', JSON.stringify(this.currentPosition));
      }
    };

    this.overlay.addEventListener('pointerdown', onPointerDown);

    // Reposition on window resize
    window.addEventListener('resize', () => this._clampPosition());
  }

  _clampPosition() {
    if (!this.currentPosition || !this.container || !this.overlay) return;
    const containerRect = this.container.getBoundingClientRect();
    const overlayRect = this.overlay.getBoundingClientRect();

    let newLeft = this.currentPosition.leftPct * containerRect.width;
    let newTop = this.currentPosition.topPct * containerRect.height;

    const minLeft = 12;
    const maxLeft = containerRect.width - overlayRect.width - 12;
    const minTop = 12;
    const maxTop = containerRect.height - overlayRect.height - 12;

    newLeft = Math.max(minLeft, Math.min(newLeft, maxLeft));
    newTop = Math.max(minTop, Math.min(newTop, maxTop));

    this.overlay.style.transform = 'none';
    this.overlay.style.bottom = 'auto';
    this.overlay.style.left = `${newLeft}px`;
    this.overlay.style.top = `${newTop}px`;
  }

  _restorePosition() {
    try {
      const saved = localStorage.getItem('wan_theater_caption_pos');
      if (saved) {
        this.currentPosition = JSON.parse(saved);
        setTimeout(() => this._clampPosition(), 100);
      }
    } catch {
      // Ignore localStorage errors
    }
  }

  resetPosition() {
    localStorage.removeItem('wan_theater_caption_pos');
    this.currentPosition = null;
    this.overlay.style.transform = 'translateX(-50%)';
    this.overlay.style.left = '50%';
    this.overlay.style.top = 'auto';
    this.overlay.style.bottom = '24px';
  }

  render(sceneData) {
    const body = this.overlay.querySelector('.caption-body');
    if (!body) return;

    const pairs = Array.isArray(sceneData?.narration_sentences) ? sceneData.narration_sentences : [];
    const isBilingual = Boolean(sceneData?.translation_language && pairs.length);

    if (!sceneData || (!sceneData.narration && !pairs.length)) {
      body.innerHTML = '<em>Listening for next scene narration…</em>';
      return;
    }

    if (!isBilingual) {
      body.textContent = sceneData.narration || '';
      return;
    }

    body.innerHTML = '';
    pairs.forEach((pair) => {
      const pairDiv = document.createElement('div');
      pairDiv.className = 'caption-bilingual-pair';

      const originalDiv = document.createElement('div');
      originalDiv.className = 'caption-original';
      originalDiv.textContent = pair.original || '';

      pairDiv.appendChild(originalDiv);

      if (pair.translation) {
        const transDiv = document.createElement('div');
        transDiv.className = 'caption-translation';
        transDiv.innerHTML = `<span class="caption-badge">Translate</span><span>${this._escapeHtml(pair.translation)}</span>`;
        pairDiv.appendChild(transDiv);
      }

      body.appendChild(pairDiv);
    });
  }

  _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }
}
