# Startbrett Private Limited — Website

Marketing website for **Startbrett Private Limited**, a Pune-based manufacturer of
control panels, industrial sensors and instrumentation, offering NABL-traceable
calibration, instrument installation and industrial/robotic automation across India.

The entire site is a **single self-contained `index.html`** — no build step, no
framework, no dependencies to install. Open it in a browser and it runs.

---

## ✨ Features

- **Single-page app** with client-side routing (Home, Services, Industries, Products,
  Projects, Case Studies, Certifications, Careers, Blog, Contact, Quote).
- **Live product catalogue** — 14 real products with photos, prices, specs and a
  same-category gallery, filterable by category.
- **Animated 3D hero** (Three.js factory scene) with a reduced-motion fallback.
- **Signature "live HMI" instrument cluster** on the hero.
- **Dark / light theme** toggle (remembered via `localStorage`).
- **Mega-menus, mobile drawer, ⌘K search overlay**, animated counters, testimonials
  carousel, before/after slider, FAQ accordions and category filters.
- **Four forms** (contact, quote, careers, newsletter) with inline validation and
  honeypot spam traps.
- **Accessibility**: keyboard-navigable, `prefers-reduced-motion` aware, semantic markup.
- **SEO**: meta tags, Open Graph and JSON-LD structured data.

## 🧱 Tech stack

- Vanilla **HTML + CSS + JavaScript** (no framework, no bundler)
- [Three.js](https://threejs.org/) (via CDN) for the hero animation
- Google Fonts — *Inter* (body) and *Space Grotesk* (headings)
- Product photos hotlinked from the company's public IndiaMART listing

## 🚀 Getting started (local)

```bash
git clone https://github.com/<your-username>/startbrett-website.git
cd startbrett-website
# just open the file:
open index.html          # macOS
# or: xdg-open index.html # Linux   |   start index.html # Windows
```

Optionally serve it over HTTP (nicer for testing):

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## 🌐 Deploy

Because it's a static `index.html`, it deploys anywhere with zero config.

**GitHub Pages**
1. Push this repo to GitHub.
2. Repo → **Settings → Pages**.
3. **Source**: `Deploy from a branch` → Branch `main` → `/ (root)` → **Save**.
4. Your site goes live at `https://<your-username>.github.io/<repo-name>/`.

**Netlify / Vercel / Cloudflare Pages** — import the repo and deploy; no build command,
publish directory `/` (root).

## 🗂️ Structure

```
startbrett-website/
├── index.html      # the entire website (HTML + CSS + JS inline)
├── README.md
├── LICENSE
└── .gitignore
```

## ✅ Before you go live — customization checklist

A few values are placeholders or come from the public IndiaMART profile and should be
finalized with real company data:

- [ ] **Direct phone number.** The listed number (`+91 80477 85654`) is an IndiaMART
      relay line. Replace with the company's direct line. Search the file for
      `80477 85654` and `tel:+918047785654`.
- [ ] **Email.** `sales@startbrett.com` is a placeholder — the real address isn't public.
- [ ] **Full address / exact GST & CIN.** Only "Pune, Maharashtra" and masked IDs are
      public; fill in the full details (search for `U31109PN2017PTC`, `27**********1ZV`).
- [ ] **Testimonials** are generic, role-based placeholders — swap in real client quotes
      (search for `const TESTI`).
- [ ] **Projects & Case Studies** are clearly labelled *representative/illustrative*
      examples; replace the metrics with genuine project details or keep the disclaimer.
- [ ] **Product images** are hotlinked from IndiaMART. For reliability, download them and
      host locally in an `assets/` folder, then update the `IMG` base URL in the
      `PRODUCTS` array.
- [ ] **Forms are front-end only** — they validate but don't send. Wire them to a service
      like [Formspree](https://formspree.io/), Netlify Forms, or your own endpoint.
- [ ] Add real **Open Graph / favicon images** and set the canonical URL.

## 📄 License

See [LICENSE](LICENSE). This is proprietary company material by default — change the
license file if you intend to open-source it.
