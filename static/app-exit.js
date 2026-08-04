(function () {
  function exitTarget() {
    const header = document.querySelector('header');
    if (!header) return null;
    return header.querySelector('nav') || header.lastElementChild || header;
  }

  function overlay() {
    const element = document.createElement('div');
    element.id = 'appExitOverlay';
    element.innerHTML = '<div><strong>Archiving and releasing local resources…</strong><p>Active work is being kept resumable. Gemma, Supertonic, FFmpeg, and app-owned ComfyUI processes are stopping.</p></div>';
    document.body.appendChild(element);
    return element;
  }

  async function exitApp(button) {
    button.disabled = true;
    const screen = overlay();
    try {
      const response = await fetch('/api/shutdown', {
        method: 'POST',
        headers: {'X-Wan-Local-Exit': 'release-owned-resources'},
      });
      if (!response.ok) throw new Error(await response.text() || 'Could not shut down the app.');
      const started = Date.now();
      while (Date.now() - started < 30000) {
        await new Promise(resolve => setTimeout(resolve, 500));
        try {
          await fetch('/api/status', {cache: 'no-store'});
        } catch {
          screen.innerHTML = '<div><strong>Everything is safely closed.</strong><p>The local server stopped and app-owned RAM/VRAM reservations were released. You can close this tab.</p></div>';
          return;
        }
      }
      screen.innerHTML = '<div><strong>Shutdown is taking longer than expected.</strong><p>Your completed work is archived. The remaining local process is still being released.</p></div>';
    } catch (error) {
      screen.remove();
      button.disabled = false;
      button.textContent = 'Exit and release';
      window.alert(error.message || 'Could not shut down the app.');
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    const target = exitTarget();
    if (!target) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'app-exit-button';
    button.textContent = 'Exit and release';
    button.title = 'Archive active work, release RAM/VRAM, and close the local server';
    button.addEventListener('click', function () { exitApp(button); });
    target.appendChild(button);
  });
})();
