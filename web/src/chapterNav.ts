/** Shared prev/next + chapter strip for the five-page tour. */

export type NavLink = { href: string; label: string };

const STRIP: { href: string; label: string; n: 1 | 2 | 3 | 4 | 5 }[] = [
  { n: 1, href: "./", label: "Ellipse" },
  { n: 2, href: "./truncate.html", label: "Truncate" },
  { n: 3, href: "./faces.html", label: "Faces" },
  { n: 4, href: "./gradient.html", label: "Gradient" },
  { n: 5, href: "./noise.html", label: "Sampling" },
];

export function chapterNav(opts: {
  prev?: NavLink;
  next?: NavLink;
  current: 1 | 2 | 3 | 4 | 5;
}): string {
  const parts: string[] = [];
  if (opts.prev) {
    parts.push(`<a href="${opts.prev.href}">${opts.prev.label}</a>`);
  }
  if (opts.next) {
    if (parts.length) parts.push("·");
    parts.push(`<a href="${opts.next.href}">${opts.next.label}</a>`);
  }
  const strip = STRIP.map((s) => {
    const cur = s.n === opts.current ? ' aria-current="page"' : "";
    return `<a href="${s.href}"${cur}>${s.label}</a>`;
  }).join('<span class="strip-sep" aria-hidden="true">·</span>');

  return `
    <p class="chapter-nav">${parts.join("\n      ") || "&nbsp;"}</p>
    <nav class="chapter-strip" aria-label="Chapters">${strip}</nav>
  `;
}
