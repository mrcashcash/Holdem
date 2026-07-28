import { execFile } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { promisify } from "node:util";

const outputDirectory = resolve("frontend/public/assets/casino-cards");
const ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"];
const suits = [
  { code: "s", symbol: "♠", name: "spades", light: "#9aa2a9", mid: "#424950", dark: "#10151b" },
  { code: "h", symbol: "♥", name: "hearts", light: "#e98486", mid: "#b83845", dark: "#5f0713" },
  { code: "d", symbol: "♦", name: "diamonds", light: "#69bde7", mid: "#2184bd", dark: "#075282" },
  { code: "c", symbol: "♣", name: "clubs", light: "#7ed679", mid: "#33ad47", dark: "#087125" },
];

const execFileAsync = promisify(execFile);

async function loadArialBlackGlyphs() {
  if (process.platform !== "win32") return null;
  const script = String.raw`
Add-Type -AssemblyName PresentationCore
$fontPath = (Join-Path $env:WINDIR 'Fonts\ariblk.ttf').Replace('\', '/')
$fontUri = [Uri]::new("file:///$fontPath")
$typeface = [System.Windows.Media.GlyphTypeface]::new($fontUri)
$glyphs = [ordered]@{}
foreach ($character in @('A', '2', '3', '4', '5', '6', '7', '8', '9', '1', '0', 'J', 'Q', 'K')) {
  $glyphIndex = $typeface.CharacterToGlyphMap[[int][char]$character]
  $outline = $typeface.GetGlyphOutline($glyphIndex, 1.0, 1.0)
  $glyphs[$character] = [ordered]@{
    advance = [double]$typeface.AdvanceWidths[$glyphIndex]
    path = $outline.ToString([Globalization.CultureInfo]::InvariantCulture).Substring(2)
  }
}
$glyphs | ConvertTo-Json -Compress -Depth 4
`;
  const { stdout } = await execFileAsync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { maxBuffer: 1024 * 1024 },
  );
  return JSON.parse(stdout.trim());
}

// Poker logic and asset names keep using "T", while the face follows the
// conventional, screen-readable "10" shown in the table reference.
const displayRank = (rank) => (rank === "T" ? "10" : rank);
const escapeXml = (value) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const cardGlyphs = await loadArialBlackGlyphs();

function rankArtwork(
  shownRank,
  {
    centerX,
    leftX,
    baselineY,
    fontSize,
    letterSpacing,
    scaleX = 1,
  },
) {
  if (!cardGlyphs) {
    const alignment = leftX === undefined ? ` text-anchor="middle"` : "";
    return `<text x="${leftX ?? centerX}" y="${baselineY}"${alignment} font-family="'Arial Black', Arial, sans-serif" font-size="${fontSize}" font-weight="900" letter-spacing="${letterSpacing}">${escapeXml(shownRank)}</text>`;
  }

  const glyphs = [...shownRank].map((character) => cardGlyphs[character]);
  const unscaledWidth = glyphs.reduce(
    (total, glyph, index) =>
      total + glyph.advance * fontSize + (index < glyphs.length - 1 ? letterSpacing : 0),
    0,
  );
  const startX = leftX ?? centerX - (unscaledWidth * scaleX) / 2;
  let cursor = 0;
  return glyphs
    .map((glyph, index) => {
      const translateX = startX + cursor * scaleX;
      cursor += glyph.advance * fontSize;
      if (index < glyphs.length - 1) cursor += letterSpacing;
      const matrix = [
        fontSize * scaleX,
        0,
        0,
        fontSize,
        translateX,
        baselineY,
      ].map((value) => Number(value.toFixed(6))).join(" ");
      return `<path d="${glyph.path}" transform="matrix(${matrix})"/>`;
    })
    .join("");
}

function cardSvg(rank, suit, { x = 0, y = 0, scale = 1, id }) {
  const shownRank = displayRank(rank);
  const isTen = rank === "T";
  const cornerFontSize = isTen ? 38 : 40;
  const cornerLetterSpacing = isTen ? -3 : -1;
  const largeFontSize = isTen ? 154 : 200;
  const largeLetterSpacing = isTen ? -9 : -7;
  const largeScaleX = isTen ? 1 : 1.14;
  const rx = 14;
  const shadowId = `shadow-${id}`;
  const gradientId = `gradient-${id}`;
  const shineId = `shine-${id}`;
  const glowId = `glow-${id}`;
  const textShadowId = `text-shadow-${id}`;
  return `<g transform="translate(${x} ${y}) scale(${scale})" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
    <defs>
      <linearGradient id="${gradientId}" x1="0" y1="0" x2=".72" y2="1"><stop stop-color="${suit.dark}"/><stop offset=".3" stop-color="${suit.light}"/><stop offset=".58" stop-color="${suit.mid}"/><stop offset="1" stop-color="${suit.dark}"/></linearGradient>
      <linearGradient id="${shineId}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#fff" stop-opacity=".42"/><stop offset=".12" stop-color="#fff" stop-opacity=".12"/><stop offset=".39" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <radialGradient id="${glowId}" cx="76%" cy="4%" r="66%"><stop stop-color="#fff" stop-opacity=".22"/><stop offset=".48" stop-color="#fff" stop-opacity="0"/></radialGradient>
      <filter id="${shadowId}" x="-20%" y="-20%" width="145%" height="150%"><feDropShadow dx="4" dy="6" stdDeviation="4" flood-color="#000" flood-opacity=".58"/></filter>
      <filter id="${textShadowId}" x="-25%" y="-25%" width="150%" height="155%"><feDropShadow dx="1.5" dy="2.5" stdDeviation="1.2" flood-color="#000" flood-opacity=".7"/></filter>
    </defs>
    <rect x="0" y="0" width="250" height="360" rx="${rx}" fill="#0a0e14" filter="url(#${shadowId})"/>
    <rect x="6" y="6" width="238" height="348" rx="10" fill="url(#${gradientId})" stroke="#05080b" stroke-width="3"/>
    <rect x="9" y="9" width="232" height="342" rx="8" fill="none" stroke="#fff" stroke-opacity=".17" stroke-width="1.5"/>
    <rect x="10" y="10" width="230" height="340" rx="7" fill="url(#${glowId})"/>
    <rect x="10" y="10" width="230" height="340" rx="7" fill="url(#${shineId})"/>
    <g fill="#fff" filter="url(#${textShadowId})">
      ${rankArtwork(shownRank, {
        leftX: 24,
        baselineY: 58,
        fontSize: cornerFontSize,
        letterSpacing: cornerLetterSpacing,
      })}
      <text x="27" y="119" font-family="Georgia, 'Times New Roman', serif" font-size="30">${suit.symbol}</text>
      ${rankArtwork(shownRank, {
        centerX: 125,
        baselineY: 286,
        fontSize: largeFontSize,
        letterSpacing: largeLetterSpacing,
        scaleX: largeScaleX,
      })}
    </g>
  </g>`;
}

await mkdir(outputDirectory, { recursive: true });
for (const suit of suits) {
  for (const rank of ranks) {
    const body = cardSvg(rank, suit, { id: `${rank}${suit.code}` });
    await writeFile(`${outputDirectory}/${rank}${suit.code}.svg`, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 250 360">${body}</svg>\n`);
  }
}

const sheetWidth = 13 * 250 + 12 * 20 + 40;
const sheetHeight = 4 * 360 + 3 * 20 + 40;
const sheet = suits.flatMap((suit, row) => ranks.map((rank, column) => cardSvg(rank, suit, {
  x: 20 + column * 270,
  y: 20 + row * 380,
  id: `sheet-${rank}${suit.code}`,
}))).join("\n");
await writeFile(`${outputDirectory}/casino-deck-52.svg`, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${sheetWidth} ${sheetHeight}" width="${sheetWidth}" height="${sheetHeight}"><rect width="100%" height="100%" fill="#10151c"/>${sheet}</svg>\n`);

console.log(`Generated ${ranks.length * suits.length} cards plus casino-deck-52.svg in ${outputDirectory}`);
