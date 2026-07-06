# Screenshots & Videos

This directory hosts README-facing visual assets. **All files here are
referenced from the root `README.md`** — when you add a new asset, update
README.md to point at it.

## Current files

| File | Type | Size | Purpose |
|---|---|---|---|
| `icon.png` | Static PNG, 512x512 | ~250 KB | README hero image — the whicc app icon |
| `demo.mov` | QuickTime H.264 1080p60 | ~13 MB | 38-second live demo video, embedded via `<video>` tag |

If you want per-feature screenshots later (e.g. `settings.png`,
`glossary.png`), follow the `<feature>.png` convention.

## File convention (general)

| Asset type | Recommended size | Format | Naming |
|---|---|---|---|
| App icon / hero | ≤ 500 KB | PNG | `icon.png` |
| Static screenshot | ≤ 600 KB | PNG | `demo.png`, `<feature>.png` |
| Animation (loop) | ≤ 1.5 MB | GIF | `demo.gif` (≤ 15s at 15fps) |
| Live demo video | ≤ 25 MB | QuickTime `.mov` | `demo.mov` |

GitHub README size limits:
- Images render up to **10 MB**
- Videos in `<video>` tags render up to about **20 MB** in practice (above
  that, the asset may not load inline — push to Releases instead)

## Hero icon

`icon.png` is the README's first visual. Source from the AppIcon:

```bash
# 512x512 PNG from iconset (best balance of detail vs file size)
cp macui/Resources/AppIcon.iconset/icon_512x512.png docs/screenshots/icon.png
```

If you ever redesign the icon, regenerate this file from the new iconset
in the same way. The icon should be visually distinct at 64x64 (it'll be
displayed at ~240px in README so 512 is enough headroom).

## Video (`<video>` tag)

GitHub renders HTML5 `<video>` inline in README, with autoplay disabled
(must click play). 30-60s clips work fine; longer is OK too as long as
file size stays under 20 MB.

**How to record:**
1. Run `whicc.app` over a foreign-language video
2. `⌘⇧5` → "Record Selected Portion" → record the segment
3. Verify size with `du -h demo.mov` — keep under 20 MB
4. **Trim** if needed (raw recordings often include pre-roll):
   ```bash
   ffmpeg -i raw.mov -ss 2 -t 38 -c copy demo.mov
   ```
5. Place at `docs/screenshots/demo.mov`

**README syntax:**
```markdown
<video src="docs/screenshots/demo.mov" controls width="720" preload="metadata"></video>
```

- `controls` — shows play/pause UI
- `preload="metadata"` — loads duration only, defers video data until
  click (faster page load than `preload="auto"`)
- No `autoplay` — README visitors don't expect sound blasting on page load

## Static screenshot (when you need one)

1. Run `whicc.app`
2. Open any foreign-language video (YouTube / 直播 etc.)
3. `⌘⇧4` → drag region over the subtitle area
4. Crop / compress with `sips -Z 1600 file.png` (down to 1600px max
   dimension)
5. Save as `docs/screenshots/<feature>.png`

## Don't commit

- Personal / identifying content (browser tabs, bookmarks)
- Personal LAN URLs (e.g. `http://192.168.1.42:1234`) — the screenshot
  should show generic placeholder URLs
- **Large video files** (>25 MB) — host on GitHub Releases / YouTube
  instead; a giant `demo.mov` bloats `git clone` for every contributor