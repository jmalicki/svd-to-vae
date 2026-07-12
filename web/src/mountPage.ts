/** Inject chapter markup from `*.html?raw` with nav (and optional base URL) placeholders. */

export function mountPage(
  app: HTMLElement,
  html: string,
  opts: { nav: string; baseUrl?: string },
): void {
  let out = html.replaceAll("{{CHAPTER_NAV}}", opts.nav);
  if (opts.baseUrl !== undefined) {
    out = out.replaceAll("{{BASE_URL}}", opts.baseUrl);
  }
  app.innerHTML = out;
}
