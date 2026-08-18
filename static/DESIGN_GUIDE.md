# Endless Theater — YouTube-Native Design Guide

Welcome to the **YouTube-Native Endless Theater** design system. If someone only knows YouTube, this interface feels 100% like home: familiar layout, 100vh zero outer scroll, standard YouTube search prompt bar, video controls, draggable Closed Captions (CC), and an "Up Next" / Live Chat sidebar.

---

## 📁 Modular Stylesheet Architecture

All visual styling is organized in `static/css/`:

```text
static/
├── css/
│   ├── tokens.css             <-- 🎨 YOUTUBE PALETTE: Backgrounds, red accents, CC captions
│   ├── layout.css             <-- 📐 100VH WATCH LAYOUT: Fixed header, 2-column zero-scroll grid
│   ├── youtube-player.css     <-- 🎬 VIDEO CONTROLS: Scrubber, Play/Pause, Live chip, CC
│   ├── subtitles.css          <-- 💬 YOUTUBE CC CAPTIONS: Draggable caption overlay & translations
│   ├── youtube-sidebar.css    <-- 📑 SIDEBAR: Recommendation cards, Live Chat stream, filter chips
│   ├── components.css         <-- 🔘 CHANNEL ROW & MODAL: Action buttons, Description, Settings
│   ├── animations.css         <-- ✨ MOTION: Stream waves, spinner
│   └── index.css              <-- Main aggregator entrypoint
├── js/
│   ├── subtitles.js           <-- Draggable caption mechanics
│   └── theater.js             <-- Search prompt bar, player controls, polling
└── index.html                 <-- Semantic YouTube watch-page markup
```

---

## 🎨 Fast Theming via `tokens.css`

To modify the theme, open `static/css/tokens.css` and edit the CSS variables:

### 1. YouTube Brand Accents
```css
--yt-red: #ff0000;                     /* YouTube Brand Red */
--yt-blue: #3ea6ff;                    /* YouTube link & button blue */
--yt-white: #f1f1f1;                   /* Primary white text */
--yt-text-secondary: #aaaaaa;          /* Secondary metadata text */
```

### 2. Surfaces & Backgrounds
```css
--yt-bg-base: #0f0f0f;                 /* Main watch page background */
--yt-surface-desc: #272727;            /* Description card background */
--yt-surface-input: #121212;           /* Search & chat input background */
```

### 3. Real-Time Draggable Closed Captions (CC)
```css
--yt-cc-bg: rgba(8, 8, 8, 0.78);       /* YouTube CC semi-transparent dark background */
--yt-cc-text: #ffffff;                 /* High contrast subtitle text */
--yt-cc-font-size: 17px;               /* Subtitle text size */
--yt-cc-translation: #3ea6ff;          /* Translation line color */
```

---

## 💬 Real-Time Draggable Closed Captions

- Spoken narration floats directly over the video canvas in a classic YouTube CC box.
- Users can click and drag the caption box anywhere on the screen (e.g. top-left, center, bottom-right).
- Positions are clamped within the video stage and saved to `localStorage`.
- Hovering over the captions reveals a **Reset** button to snap back to bottom-center.

---

## 🛡️ Safe Styling Rules

1. ✅ **Edit CSS freely** in `static/css/*.css`. Refresh `http://127.0.0.1:7868/` to see your changes instantly.
2. ⚠️ **Keep element IDs intact** in `index.html` (such as `id="ytSearchForm"`, `id="ytPlayerContainer"`, `id="playerA"`, `id="ytCaptionWindow"`), as JavaScript binds to these IDs.
