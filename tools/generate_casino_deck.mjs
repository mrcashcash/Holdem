import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const outputDirectory = resolve("frontend/public/assets/casino-cards");
const ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"];
const suits = [
  { code: "s", symbol: "♠", name: "spades", light: "#89939d", mid: "#3f464d", dark: "#11161c" },
  { code: "h", symbol: "♥", name: "hearts", light: "#e98486", mid: "#b83845", dark: "#5f0713" },
  { code: "d", symbol: "♦", name: "diamonds", light: "#67b6de", mid: "#1d79ad", dark: "#06436f" },
  { code: "c", symbol: "♣", name: "clubs", light: "#70c56d", mid: "#29923c", dark: "#075d24" },
];

const displayRank = (rank) => rank;
const escapeXml = (value) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function cardSvg(rank, suit, { x = 0, y = 0, scale = 1, id }) {
  const shownRank = displayRank(rank);
  const rx = 19;
  const shadowId = `shadow-${id}`;
  const gradientId = `gradient-${id}`;
  const shineId = `shine-${id}`;
  const glowId = `glow-${id}`;
  const textShadowId = `text-shadow-${id}`;
  return `<g transform="translate(${x} ${y}) scale(${scale})">
    <defs>
      <linearGradient id="${gradientId}" x1="0" y1="0" x2="0.72" y2="1"><stop stop-color="${suit.dark}"/><stop offset=".26" stop-color="${suit.light}"/><stop offset=".54" stop-color="${suit.mid}"/><stop offset="1" stop-color="${suit.dark}"/></linearGradient>
      <linearGradient id="${shineId}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#fff" stop-opacity=".52"/><stop offset=".1" stop-color="#fff" stop-opacity=".16"/><stop offset=".43" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <radialGradient id="${glowId}" cx="78%" cy="5%" r="68%"><stop stop-color="#fff" stop-opacity=".35"/><stop offset=".5" stop-color="#fff" stop-opacity="0"/></radialGradient>
      <filter id="${shadowId}" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="4" dy="6" stdDeviation="4" flood-color="#000" flood-opacity=".48"/></filter>
      <filter id="${textShadowId}" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="2" dy="3" stdDeviation="1.5" flood-color="#000" flood-opacity=".72"/></filter>
    </defs>
    <rect x="0" y="0" width="250" height="360" rx="${rx}" fill="#0a0e14" filter="url(#${shadowId})"/>
    <rect x="5" y="5" width="240" height="350" rx="15" fill="url(#${gradientId})" stroke="#05080b" stroke-width="3"/>
    <rect x="9" y="9" width="232" height="342" rx="12" fill="none" stroke="#fff" stroke-opacity=".25" stroke-width="2"/>
    <rect x="10" y="10" width="230" height="340" rx="11" fill="url(#${glowId})"/>
    <rect x="10" y="10" width="230" height="340" rx="11" fill="url(#${shineId})"/>
    <g fill="#fff" filter="url(#${textShadowId})">
      <text x="24" y="70" font-family="Arial Black, Arial, sans-serif" font-size="56" font-weight="900" letter-spacing="-3">${escapeXml(shownRank)}</text>
      <text x="27" y="119" font-family="Georgia, 'Times New Roman', serif" font-size="45">${suit.symbol}</text>
      <text x="125" y="298" text-anchor="middle" font-family="Arial Black, Arial, sans-serif" font-size="200" font-weight="900" letter-spacing="-7">${escapeXml(shownRank)}</text>
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
