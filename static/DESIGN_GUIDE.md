# Endless Theater — Artist & Designer Styling Guide

Welcome to the **Endless Offline Theater** styling system! This frontend is designed for visual artists and designers to rapidly adjust styles, colors, typography, and subtitle placement with **zero risk** of breaking backend Python code or frontend JavaScript logic.

---

## 📁 File Structure for Designers

All visual styling is cleanly organized in `static/css/`:

```text
static/
├── css/
│   ├── tokens.css       <-- 🎨 PRIMARY THEME FILE: Colors, fonts, shadows, glassmorphism
│   ├── subtitles.css    <-- 💬 YOUTUBE-STYLE SUBTITLE OVERLAY: Font, background, drag handle
│   ├── components.css   <-- 🔘 BUTTONS, INPUTS, CHIPS, TELEMETRY: Form styling & badges
│   ├── layout.css       <-- 📐 GRIDS & STAGE: Two-column layout, player viewport, responsiveness
│   ├── animations.css   <-- ✨ MOTION & GLOW: Keyframe pulses, spinners, fades
│   └── index.css        <-- Main stylesheet importer
├── js/
│   ├── subtitles.js     <-- Draggable caption mechanics (don't touch unless adding JS features)
│   └── theater.js       <-- Application streaming logic
└── index.html           <-- Semantic HTML5 structure
```

---

## 🎨 Fast Theming with Design Tokens (`tokens.css`)

To change the look and feel of the entire application, simply open `static/css/tokens.css` and edit the CSS variables.

### 1. Brand Accents
```css
--color-brand-primary: #38bdf8;     /* Primary interactive glow (buttons, highlights) */
--color-brand-secondary: #818cf8;   /* Companion accent glow */
--color-brand-gradient: linear-gradient(135deg, #38bdf8, #818cf8 50%, #c084fc);
--color-brand-glow: rgba(56, 189, 248, 0.25);
```

### 2. Canvas & Surface Tones
```css
--bg-app: #08090d;                    /* App background canvas */
--surface-card: rgba(17, 20, 29, 0.85);/* Frosted card surface */
--border-subtle: rgba(255, 255, 255, 0.07); /* Clean 1px border */
```

### 3. Real-Time YouTube-Style Subtitles
```css
--caption-bg: rgba(9, 10, 15, 0.88);  /* Caption chip background */
--caption-blur: blur(16px);           /* Glassmorphism blur */
--caption-text-color: #ffffff;        /* High-contrast subtitle text */
--caption-translation-color: #67e8f9; /* Translation subtitle highlight */
--caption-radius: 14px;               /* Caption pill roundedness */
--caption-font-size: 16px;            /* Subtitle text size */
```

---

## 🌈 Preset Theme Recipes

Copy and paste any of these recipes into `:root` in `static/css/tokens.css` to instantly transform the theater:

### 🌟 1. Google DeepMind Minimalist (Default)
```css
--color-brand-primary: #38bdf8;
--color-brand-secondary: #818cf8;
--color-brand-gradient: linear-gradient(135deg, #38bdf8, #818cf8 50%, #c084fc);
--color-brand-glow: rgba(56, 189, 248, 0.25);
--bg-app: #08090d;
--surface-card: rgba(17, 20, 29, 0.85);
```

### 🎬 2. Warm Cinema 35mm
```css
--color-brand-primary: #f59e0b;
--color-brand-secondary: #f97316;
--color-brand-gradient: linear-gradient(135deg, #f59e0b, #f97316 60%, #ef4444);
--color-brand-glow: rgba(245, 158, 11, 0.3);
--bg-app: #0a0908;
--surface-card: rgba(24, 20, 17, 0.9);
--caption-translation-color: #fde68a;
```

### 🌌 3. Cyberpunk Synthwave
```css
--color-brand-primary: #ec4899;
--color-brand-secondary: #8b5cf6;
--color-brand-gradient: linear-gradient(135deg, #ec4899, #8b5cf6 50%, #06b6d4);
--color-brand-glow: rgba(236, 72, 153, 0.35);
--bg-app: #07050f;
--surface-card: rgba(19, 14, 33, 0.88);
--caption-translation-color: #a5f3fc;
```

### 🌿 4. Emerald Studio Minimalist
```css
--color-brand-primary: #10b981;
--color-brand-secondary: #06b6d4;
--color-brand-gradient: linear-gradient(135deg, #10b981, #06b6d4);
--color-brand-glow: rgba(16, 185, 129, 0.25);
--bg-app: #060c09;
--surface-card: rgba(11, 23, 18, 0.9);
--caption-translation-color: #6ee7b7;
```

---

## 💬 Customizing YouTube-Inspired Real-Time Subtitles

The real-time subtitles overlay (`#captionOverlay`) floats over the video stage and can be dragged anywhere by the user.

- **Background & Acrylic Blur**: Edit `--caption-bg` and `--caption-blur` in `tokens.css`.
- **Drag Handle**: In `subtitles.css`, edit `.caption-drag-pill` to change the grab handle size, color, or pill thickness.
- **Bilingual Tag Badge**: In `subtitles.css`, edit `.caption-badge` to customize the `[Translate]` tag styling.
- **Position Persistence**: The subtitle position is clamped automatically within video bounds and saved in `localStorage`. Clicking **Reset** restores it to bottom-center.

---

## 🛡️ Safe Styling Rules of Thumb

1. ✅ **DO edit CSS files** (`static/css/*.css`) freely. Refresh your browser at `http://127.0.0.1:7868/` to see your changes immediately.
2. ✅ **DO change layout structure or card padding** in `layout.css`.
3. ⚠️ **DON'T delete element IDs** in `index.html` (such as `id="screen"`, `id="playerA"`, `id="captionOverlay"`, `id="start"`), as JavaScript binds to these IDs.
