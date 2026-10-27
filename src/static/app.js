const $ = id => document.getElementById(id);
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.text()).trim() || response.statusText);
  return response.status === 204 ? null : response.json();
};
const COLORS = {
  title: '#d97706', section: '#b45309', text: '#2563eb', list: '#0891b2', caption: '#c026d3',
  footnote: '#ea580c', aside_text: '#4f46e5', header: '#78716c', footer: '#78716c', page_number: '#57534e',
  table: '#dc2626', chart: '#db2777', formula: '#0d9488', code: '#16a34a', image: '#7c3aed',
};
const colorOf = label => COLORS[label] || '#9ca3af';
const active = job => job.status === 'queued' || job.status === 'running';
const state = {
  runs: [], jobs: [], run: null, page: 0, zoom: 1, view: 'page', selected: null,
  hidden: new Set(), file: null, follow: null, endpoint: null,
  maxUploadBytes: 256 * 1024 * 1024, content: new Map(), continuing: new Set(),
};

async function refreshSidebar() {
  [state.runs, state.jobs] = await Promise.all([api('/api/runs'), api('/api/jobs')]);
  renderSidebar();
  const followed = state.jobs.find(job => job.id === state.follow);
  if (followed && !active(followed)) {
    state.follow = null;
    if (followed.status === 'done') {
      await openRun(followed.name);
    } else {
      toast(new Error(followed.error || 'Document parsing failed; partial results are retained.'));
    }
  }
  if (state.jobs.some(active)) watchJobs();
}

function renderSidebar() {
  const jobs = state.jobs.filter(job => job.status !== 'done');
  const runs = state.runs.filter(run => !jobs.some(job => active(job) && job.name === run.name));
  $('runs-count').textContent = runs.length || '';
  $('runs').replaceChildren(...jobs.map(jobRow), ...runs.map(runRow));
  if (!$('runs').children.length) {
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = 'No saved documents yet.';
    $('runs').appendChild(hint);
  }
}

function runRow(run) {
  const row = document.createElement('div');
  row.className = 'run' + (state.run?.name === run.name ? ' current' : '');
  const link = document.createElement('a');
  link.className = 'run-name';
  link.href = '#' + encodeURIComponent(run.name);
  link.textContent = run.name;
  const meta = document.createElement('span');
  meta.className = 'run-meta';
  meta.textContent = `${run.page_count} pages · ${run.status} · ${when(run.modified)}`;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'delete';
  button.textContent = '×';
  button.setAttribute('aria-label', `Delete ${run.name}`);
  button.onclick = () => deleteRun(button, run.name);
  row.append(link, meta, button);
  return row;
}

function jobRow(job) {
  const row = document.createElement('div');
  row.className = `run job ${job.status}`;
  row.setAttribute('aria-label', `${job.name}: ${job.status}`);
  const name = document.createElement('span');
  name.className = 'run-name';
  name.textContent = job.name;
  const meta = document.createElement('span');
  meta.className = 'run-meta';
  meta.textContent = job.status === 'error' ? job.error : job.status === 'queued' ? 'Queued' :
    job.total ? `${job.done} / ${job.total} pages` : 'Preparing document…';
  row.append(name, meta);
  if (active(job)) {
    const progress = document.createElement('progress');
    progress.setAttribute('aria-label', `Parsing ${job.name}`);
    if (job.total) { progress.max = job.total; progress.value = job.done; }
    row.appendChild(progress);
  } else {
    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'delete';
    dismiss.textContent = '×';
    dismiss.setAttribute('aria-label', `Dismiss job ${job.name}`);
    dismiss.onclick = () => api('/api/jobs/' + job.id, { method: 'DELETE' }).then(refreshSidebar).catch(toast);
    row.appendChild(dismiss);
  }
  return row;
}

function deleteRun(button, name) {
  if (!button.classList.contains('armed')) {
    button.classList.add('armed');
    button.textContent = 'Delete?';
    setTimeout(() => { button.classList.remove('armed'); button.textContent = '×'; }, 3000);
    return;
  }
  api('/api/runs/' + encodeURIComponent(name), { method: 'DELETE' })
    .then(() => { if (state.run?.name === name) closeRun(); return refreshSidebar(); })
    .catch(toast);
}

function when(seconds) {
  const age = Date.now() / 1000 - seconds;
  if (age < 60) return 'just now';
  if (age < 3600) return `${Math.floor(age / 60)} min ago`;
  if (age < 86400) return `${Math.floor(age / 3600)}h ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined, { dateStyle: 'medium' });
}

let poller = null;
let polling = false;
function watchJobs() {
  if (poller) return;
  poller = setInterval(async () => {
    if (polling) return;
    polling = true;
    try { await refreshSidebar(); }
    catch (error) { toast(error); }
    finally { polling = false; }
    if (!state.jobs.some(active)) { clearInterval(poller); poller = null; }
  }, 800);
}

let openRequest = 0;
async function openRun(name) {
  const request = ++openRequest;
  const run = await api('/api/runs/' + encodeURIComponent(name));
  if (request !== openRequest) return;
  Object.assign(state, { run, page: 0, zoom: 1, view: 'page', selected: null,
    content: new Map(run.blocks.map(block => [block.id, block])),
    continuing: new Set(run.continuations.map(link => link.to)),
  });
  state.hidden.clear();
  $('left').scrollTop = $('output-scroll').scrollTop = 0;
  if (decodeURIComponent(location.hash.slice(1)) !== name) history.replaceState(null, '', '#' + encodeURIComponent(name));
  document.body.classList.remove('menu-open');
  $('menu').setAttribute('aria-expanded', 'false');
  renderSidebar();
  renderViewer();
}

function closeRun() {
  openRequest++;
  state.run = null;
  history.replaceState(null, '', location.pathname);
  renderSidebar();
  renderViewer();
}

function renderViewer() {
  const run = state.run;
  document.title = run ? `${run.name} · Folio` : 'Folio documents';
  document.querySelector('.toolbar').classList.toggle('idle', !run);
  $('title').textContent = run ? run.name : 'Document viewer';
  $('viewer').hidden = !run;
  $('empty').hidden = !!run;
  $('page-stack').replaceChildren();
  $('block-inspector').hidden = true;
  if (!run) return;
  renderPages();
  renderOutput();
}

function positionBox(element, bbox) {
  const [x1, y1, x2, y2] = bbox;
  Object.assign(element.style, { left: `${x1 / 10}%`, top: `${y1 / 10}%`, width: `${(x2 - x1) / 10}%`, height: `${(y2 - y1) / 10}%` });
}

function renderPages() {
  const run = state.run;
  run.pages.forEach((result, index) => {
    const section = document.createElement('section');
    section.className = 'physical-page';
    section.dataset.pageIndex = index;
    const heading = document.createElement('h2');
    const select = document.createElement('button');
    select.type = 'button';
    select.className = 'page-title';
    select.textContent = `Page ${index + 1}`;
    select.setAttribute('aria-label', `Select page ${index + 1}`);
    select.onclick = () => goTo(index, false, state.view === 'document' ? 'page' : state.view);
    heading.appendChild(select);
    const page = document.createElement('div');
    page.className = 'page';
    page.style.aspectRatio = `${result.width} / ${result.height}`;
    const image = document.createElement('img');
    image.src = pageImage(index);
    image.alt = '';
    image.loading = 'lazy';
    positionBox(image, [0, 0, 1000, 1000]);
    page.appendChild(image);
    result.blocks.forEach((placement, blockIndex) => {
      const block = state.content.get(placement.block_id), bbox = placement.bbox;
      if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isFinite) || bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) return;
      const box = document.createElement('div');
      box.className = 'box' + (state.continuing.has(placement.id) ? ' cont' : '');
      box.dataset.i = `${index}:${blockIndex}`;
      box.dataset.pageIndex = index;
      box.dataset.blockId = block.id;
      box.dataset.label = block.label;
      box.tabIndex = 0;
      box.setAttribute('role', 'button');
      box.setAttribute('aria-label', `Page ${index + 1}, ${block.label}, block ${blockIndex + 1}`);
      box.style.setProperty('--c', colorOf(block.label));
      positionBox(box, bbox);
      page.appendChild(box);
    });
    section.append(heading, page);
    $('page-stack').appendChild(section);
  });
  fitPage();
}

function renderOutput() {
  const run = state.run;
  if (!run) return;
  const documentOutput = run.document;
  const markdown = documentOutput.markdown;
  $('view').querySelector('[value=document]').disabled = markdown == null;
  if (state.view === 'document' && markdown == null) state.view = 'page';
  $('view').value = state.view;
  $('download').disabled = !documentOutput.download;
  const pages = run.pages, p = pages[state.page];
  $('pageno').value = p ? state.page + 1 : '';
  $('pageno').max = Math.max(1, pages.length);
  $('pageno').disabled = !p;
  $('count').textContent = `/ ${pages.length}`;
  $('page-label').textContent = p ? `Page ${state.page + 1}` : 'No rendered pages';
  $('prev').disabled = state.page === 0 || !p;
  $('next').disabled = state.page >= pages.length - 1;
  $('page-stack').querySelectorAll('.physical-page').forEach(page => {
    const selected = Number(page.dataset.pageIndex) === state.page;
    page.classList.toggle('selected', selected);
    page.querySelector('.page-title').setAttribute('aria-pressed', String(selected));
  });
  $('page-stack').querySelectorAll('.box').forEach(box => {
    box.hidden = state.hidden.has(box.dataset.label);
    box.classList.remove('inview', 'on');
  });
  $('filters').replaceChildren();
  $('type-filter').hidden = state.view !== 'page';
  $('panel').replaceChildren();
  spy.disconnect();
  if (state.view === 'document') {
    selectBlock(null);
    const doc = document.createElement('div');
    doc.className = 'doc';
    doc.appendChild(renderMarkdown(markdown || ''));
    $('panel').appendChild(doc);
    return;
  }
  if (!p) { renderEmpty('No completed pages are available. Inspect the run status and retained files.'); return; }
  if (state.view === 'raw') { selectBlock(null); renderRaw(p.raw); return; }
  const entries = pageEntries(p);
  renderFilters(entries.map(entry => entry.block));
  const reader = document.createElement('article');
  reader.className = 'page-reader';
  const heading = document.createElement('header');
  heading.className = 'reader-heading';
  const title = document.createElement('h2');
  title.textContent = `Page ${state.page + 1}`;
  const count = document.createElement('span');
  count.textContent = `${entries.filter(({block}) => !state.hidden.has(block.label)).length} blocks`;
  heading.append(title, count);
  reader.appendChild(heading);
  $('panel').appendChild(reader);
  let shown = 0;
  entries.forEach(({block, placement, index}) => {
    if (state.hidden.has(block.label)) return;
    shown++;
    const color = colorOf(block.label), label = block.label.replaceAll('_', ' ');
    const card = document.createElement('section');
    card.className = 'blk';
    card.dataset.i = `${state.page}:${index}`;
    card.dataset.pageIndex = state.page;
    card.dataset.blockId = block.id;
    card.dataset.kind = block.label;
    card.style.setProperty('--c', color);
    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = 'block-marker';
    marker.textContent = index + 1;
    marker.title = label + (state.continuing.has(placement.id) ? ' · continues' : '');
    marker.setAttribute('aria-label', `Inspect ${label}, block ${index + 1}`);
    const content = document.createElement('div');
    content.className = 'block-content';
    content.appendChild(renderBlock(block, p, placement));
    card.append(marker, content);
    reader.appendChild(card);
    spy.observe(card);
  });
  selectBlock(state.selected);
  if (!shown) renderEmpty('No visible blocks on this page.');
}

function pageEntries(page) {
  const seen = new Set();
  return page.blocks.flatMap((placement, index) => {
    if (seen.has(placement.block_id)) return [];
    seen.add(placement.block_id);
    return [{block: state.content.get(placement.block_id), placement, index}];
  });
}

function pageImage(index) {
  return `/api/runs/${encodeURIComponent(state.run.name)}/pages/${index}.png`;
}

function renderBlock(block, page, placement) {
  if (!['image', 'chart', 'header_image', 'footer_image'].includes(block.label)) return renderMarkdown(block.text || '');
  const figure = document.createElement('figure');
  figure.className = 'block-figure';
  const asset = block.assets[0] && assetUrl(block.assets[0]);
  if (asset) {
    const image = document.createElement('img');
    image.src = asset;
    image.alt = '';
    image.loading = 'lazy';
    image.className = 'asset-preview';
    figure.appendChild(image);
  } else {
    const bounds = placement.bbox;
    const preview = document.createElement('div');
    preview.className = 'image-preview';
    const ratio = (bounds[2] - bounds[0]) * page.width / ((bounds[3] - bounds[1]) * page.height);
    const decorative = block.label === 'header_image' || block.label === 'footer_image';
    preview.style.aspectRatio = ratio;
    preview.style.width = `min(100%, ${Math.min(decorative ? 100 : 420, (decorative ? 60 : 240) * ratio)}px)`;
    const image = document.createElement('img');
    image.src = pageImage(state.page);
    image.alt = '';
    image.loading = 'lazy';
    const [left, top, right, bottom] = bounds;
    positionBox(image, [-left * 1000 / (right - left), -top * 1000 / (bottom - top),
      (1000 - left) * 1000 / (right - left), (1000 - top) * 1000 / (bottom - top)]);
    preview.appendChild(image);
    figure.appendChild(preview);
  }
  const caption = document.createElement('figcaption');
  caption.appendChild(renderMarkdown(block.text || ''));
  figure.appendChild(caption);
  return figure;
}

function selectBlock(key) {
  const [pageIndex, blockIndex] = key?.split(':').map(Number) || [];
  const page = state.run?.pages?.[pageIndex], placement = page?.blocks?.[blockIndex];
  const block = placement && state.content.get(placement.block_id);
  const selected = state.view === 'page' && pageIndex === state.page && block && !state.hidden.has(block.label);
  state.selected = selected ? key : null;
  document.querySelectorAll('.box, .blk').forEach(element => {
    const matches = !!selected && element.dataset.blockId === block.id;
    element.classList.toggle('selected', matches);
    const control = element.matches('.box') ? element : element.querySelector('.block-marker');
    control?.setAttribute('aria-pressed', String(matches));
  });
  $('block-inspector').hidden = !selected;
  if (!selected) return;
  $('block-inspector').style.setProperty('--c', colorOf(block.label));
  $('inspection-type').textContent = block.label.replaceAll('_', ' ') + (state.continuing.has(placement.id) ? ' · continues' : '');
  $('inspection-position').textContent = `Block ${blockIndex + 1} · Page ${pageIndex + 1}`;
  const bbox = placement.bbox;
  $('inspection-bounds').textContent = bbox.length ? bbox.map(value => Math.round(value * 10) / 10).join(', ') : 'Unavailable';
}

function renderRaw(text) {
  const pre = document.createElement('pre');
  pre.className = 'raw';
  pre.textContent = text;
  $('panel').appendChild(pre);
}

function renderEmpty(text) {
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.textContent = text;
  $('panel').appendChild(empty);
}

function fitPage() {
  const hasPages = !!state.run?.pages?.length;
  $('zoom-out').disabled = !hasPages || state.zoom <= 1;
  $('zoom-in').disabled = !hasPages || state.zoom >= 4;
  if (!hasPages) return;
  const pane = $('left'), pad = window.innerWidth <= 600 ? 34 : 50;
  $('page-stack').querySelectorAll('.physical-page').forEach(page => {
    page.style.width = Math.max(80, Math.floor((pane.clientWidth - pad) * state.zoom)) + 'px';
  });
  $('zoom-fit').textContent = state.zoom === 1 ? 'Fit' : state.zoom.toFixed(2).replace(/\.?0+$/, '') + '×';
}

function setZoom(zoom) { state.zoom = Math.min(4, Math.max(1, zoom)); fitPage(); }

function scrollToBlock(target, pane) {
  if (!target) return;
  const box = target.getBoundingClientRect(), viewport = pane.getBoundingClientRect();
  pane.scrollTo({
    top: pane.scrollTop + box.top - viewport.top - Math.max(0, (pane.clientHeight - box.height) / 2),
    left: pane.scrollLeft + box.left - viewport.left - Math.max(0, (pane.clientWidth - box.width) / 2),
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
  });
}

function inspectBlock(target) {
  const key = target.dataset.i, fromSource = target.classList.contains('box');
  if (fromSource) goTo(Number(target.dataset.pageIndex), false, 'page');
  selectBlock(key);
  const pane = $(fromSource ? 'output-scroll' : 'left');
  scrollToBlock(pane.querySelector(fromSource ? `[data-block-id="${target.dataset.blockId}"]` : `[data-i="${key}"]`), pane);
}

function goTo(index, scrollPage = true, view = state.view) {
  const pages = state.run?.pages || [];
  if (!Number.isInteger(index) || index < 0 || index >= pages.length) return;
  const changed = index !== state.page || view !== state.view;
  if (index !== state.page) state.selected = null;
  state.page = index;
  state.view = view;
  const output = $('output-scroll'), top = output.scrollTop, left = output.scrollLeft;
  if (changed) {
    renderOutput();
    output.scrollTop = top;
    output.scrollLeft = left;
  }
  if (scrollPage) scrollToBlock($('page-stack').querySelector(`[data-page-index="${index}"]`), $('left'));
}

function renderFilters(blocks) {
  const counts = new Map();
  blocks.forEach(block => counts.set(block.label, (counts.get(block.label) || 0) + 1));
  const hidden = [...counts.keys()].filter(label => state.hidden.has(label)).length;
  $('type-count').textContent = hidden ? `${hidden} hidden` : counts.size;
  [...counts].sort((a, b) => b[1] - a[1]).forEach(([label, count], index) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.style.setProperty('--c', colorOf(label));
    chip.setAttribute('aria-pressed', String(!state.hidden.has(label)));
    chip.textContent = `${label.replaceAll('_', ' ')} ${count}`;
    chip.onclick = () => {
      state.hidden.has(label) ? state.hidden.delete(label) : state.hidden.add(label);
      renderOutput();
      $('filters').children[index]?.focus({ preventScroll: true });
    };
    $('filters').appendChild(chip);
  });
}

const boldCells = text => text.replace(/<(td|th)([^>]*)>([^<]*)<\/\1>/g,
  (match, tag, attributes, inner) => `<${tag}${attributes}>${inner.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')}</${tag}>`);

function assetUrl(path) {
  try {
    const base = new URL(state.run.document.base_url, location.origin);
    const url = new URL(path, base);
    return url.origin === location.origin && url.pathname.startsWith(base.pathname) ? url.href : null;
  } catch { return null; }
}

function renderMarkdown(text) {
  const el = document.createElement('div');
  el.className = 'md';
  if (!globalThis.marked || !globalThis.DOMPurify?.isSupported) {
    el.classList.add('text-fallback');
    el.textContent = text;
    return el;
  }
  try {
    el.appendChild(DOMPurify.sanitize(marked.parse(boldCells(text)), {
      USE_PROFILES: { html: true }, RETURN_DOM_FRAGMENT: true, ALLOW_DATA_ATTR: false,
      FORBID_TAGS: ['style', 'form', 'input', 'button', 'textarea', 'select'],
      FORBID_ATTR: ['style', 'id', 'name', 'class', 'srcset'],
    }));
    const base = new URL(state.run.document.base_url, location.origin);
    el.querySelectorAll('img').forEach(image => {
      const src = image.getAttribute('src');
      const url = src && assetUrl(src);
      if (url) image.src = url;
      else image.removeAttribute('src');
    });
    el.querySelectorAll('a[href]').forEach(link => {
      const url = new URL(link.getAttribute('href'), base);
      if (!['http:', 'https:', 'mailto:'].includes(url.protocol)) link.removeAttribute('href');
      else { link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; }
    });
    if (globalThis.renderMathInElement) renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false },
      ], throwOnError: false, trust: false,
    });
  } catch (error) {
    el.replaceChildren();
    el.classList.add('text-fallback');
    el.textContent = text;
  }
  return el;
}

const spy = new IntersectionObserver(entries => entries.forEach(entry => {
  $('page-stack').querySelectorAll(`.box[data-block-id="${entry.target.dataset.blockId}"]`).forEach(box => {
    box.classList.toggle('inview', entry.isIntersecting);
  });
}), { root: $('output-scroll'), threshold: 0.3 });
const linked = id => document.querySelectorAll(`[data-block-id="${id}"]`);
for (const [event, selected] of [['mouseover', true], ['mouseout', false]]) {
  document.addEventListener(event, e => {
    const target = e.target.closest('[data-i]');
    if (target) linked(target.dataset.blockId).forEach(el => el.classList.toggle('on', selected));
  });
}
document.addEventListener('click', e => {
  if (!e.composedPath().includes($('type-filter'))) $('type-filter').open = false;
  if (document.body.classList.contains('menu-open') && !e.target.closest('#sidebar, #menu')) {
    document.body.classList.remove('menu-open');
    $('menu').setAttribute('aria-expanded', 'false');
  }
  const target = e.target.closest('[data-i]');
  if (target && !e.target.closest('a[href]')) inspectBlock(target);
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { document.body.classList.remove('menu-open'); $('menu').setAttribute('aria-expanded', 'false'); $('type-filter').open = false; selectBlock(null); }
  if (e.target.classList.contains('box') && ['Enter', ' '].includes(e.key)) { e.preventDefault(); e.target.click(); return; }
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || e.metaKey || e.ctrlKey || document.querySelector('dialog[open]')) return;
  if (e.key === 'n') openNewRun();
  if (!state.run) return;
  if (e.key === 'ArrowLeft') goTo(state.page - 1);
  else if (e.key === 'ArrowRight') goTo(state.page + 1);
  else if (e.key === 'r') { state.view = state.view === 'raw' ? 'page' : 'raw'; $('output-scroll').scrollTop = 0; renderOutput(); }
  else if (e.key === '+' || e.key === '=') setZoom(state.zoom * 1.25);
  else if (e.key === '-') setZoom(state.zoom / 1.25);
  else if (e.key === '0') setZoom(1);
});
$('prev').onclick = () => goTo(state.page - 1);
$('next').onclick = () => goTo(state.page + 1);
$('pageno').onchange = () => { goTo(Number($('pageno').value) - 1); $('pageno').value = state.page + 1; };
$('pageno').onkeydown = event => { if (event.key === 'Enter') $('pageno').onchange(); };
$('zoom-in').onclick = () => setZoom(state.zoom * 1.25);
$('zoom-out').onclick = () => setZoom(state.zoom / 1.25);
$('zoom-fit').onclick = () => setZoom(1);
$('view').onchange = () => { state.view = $('view').value; $('output-scroll').scrollTop = 0; renderOutput(); };
$('clear-selection').onclick = () => selectBlock(null);
$('menu').onclick = () => { document.body.classList.toggle('menu-open'); $('menu').setAttribute('aria-expanded', String(document.body.classList.contains('menu-open'))); };
window.addEventListener('resize', fitPage);

function openNewRun(file = null) {
  pickFile(file);
  $('pages').value = '';
  $('form-error').hidden = true;
  if (!$('new-run-dialog').open) $('new-run-dialog').showModal();
}
function pickFile(file) {
  state.file = file || null;
  $('file-label').classList.toggle('has', !!file);
  $('file-text').textContent = file ? `${file.name} · ${(file.size / 1e6).toFixed(1)} MB` : 'Choose a PDF or image…';
}
$('download').onclick = () => { location.href = state.run.document.download; };
$('new-run').onclick = () => openNewRun();
$('cancel').onclick = () => $('new-run-dialog').close();
$('file').onchange = () => pickFile($('file').files[0]);
$('new-run-dialog').addEventListener('close', () => { $('file').value = ''; state.file = null; });
$('new-run-form').onsubmit = async e => {
  e.preventDefault();
  if (!state.file) { $('form-error').textContent = 'Choose a PDF or image first.'; $('form-error').hidden = false; return; }
  if (state.file.size > state.maxUploadBytes) {
    $('form-error').textContent = `Upload exceeds ${state.maxUploadBytes / 1024 / 1024} MiB.`;
    $('form-error').hidden = false;
    return;
  }
  const params = new URLSearchParams({pages: $('pages').value.trim()});
  $('submit').disabled = true;
  try {
    const job = await api('/api/runs?' + params, { method: 'POST', body: state.file, headers: { 'X-Filename': encodeURIComponent(state.file.name) } });
    state.follow = job.id;
    $('new-run-dialog').close();
    await refreshSidebar();
  } catch (error) { $('form-error').textContent = error.message; $('form-error').hidden = false; }
  finally { $('submit').disabled = false; }
};

const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
let dragDepth = 0;
window.addEventListener('dragenter', e => { if (hasFiles(e)) { dragDepth++; $('dropzone').hidden = false; } });
window.addEventListener('dragleave', e => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $('dropzone').hidden = true; } });
window.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('drop', e => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; $('dropzone').hidden = true;
  if (e.dataTransfer.files[0]) openNewRun(e.dataTransfer.files[0]);
});

function renderEndpoint(settings) {
  state.endpoint = settings;
  $('endpoint-url').textContent = (settings.serving ? settings.serving + ' @ ' : '') + settings.base_url;
  $('endpoint-dot').className = 'dot ' + (settings.reachable ? 'ok' : 'down');
  $('endpoint').title = settings.reachable ? `Model server online: ${settings.models.join(', ')}` : settings.error;
}
function showEndpointStatus(settings) {
  $('endpoint-status').className = 'status ' + (settings.reachable ? 'ok' : 'down');
  $('endpoint-status').textContent = settings.reachable ? `Online: ${settings.models.join(', ')}` : settings.error;
  $('api-key-status').textContent = settings.has_api_key ? 'A key is configured; leave the field blank to keep it.' : 'No API key is configured.';
}
const probeEndpoint = () => api('/api/settings').then(settings => { renderEndpoint(settings); return settings; });
function openSettings() {
  $('base-url').value = state.endpoint?.base_url || '';
  $('model-name').value = state.endpoint?.model || '';
  $('endpoint-status').className = 'status';
  $('endpoint-status').textContent = 'Checking model server…';
  $('settings-dialog').showModal();
  probeEndpoint().then(showEndpointStatus).catch(toast);
}
$('endpoint').onclick = openSettings;
$('settings-cancel').onclick = () => $('settings-dialog').close();
$('settings-dialog').addEventListener('close', () => { $('api-key').value = ''; $('clear-api-key').checked = false; });
$('settings-form').onsubmit = async e => {
  e.preventDefault();
  $('settings-save').disabled = true;
  $('endpoint-status').textContent = 'Saving and checking…';
  const body = { base_url: $('base-url').value.trim(), model: $('model-name').value.trim() };
  const key = $('api-key').value.trim();
  if (key) body.api_key = key;
  else if ($('clear-api-key').checked) body.api_key = '';
  try {
    const settings = await api('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    $('api-key').value = '';
    $('clear-api-key').checked = false;
    renderEndpoint(settings);
    showEndpointStatus(settings);
    if (settings.reachable) $('settings-dialog').close();
  } catch (error) { $('endpoint-status').className = 'status down'; $('endpoint-status').textContent = error.message; }
  finally { $('settings-save').disabled = false; }
};

let toastTimer = null;
function toast(error) {
  $('toast').textContent = error.message || String(error);
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 7000);
}
window.addEventListener('focus', () => refreshSidebar().catch(toast));
window.addEventListener('hashchange', () => {
  try {
    const name = decodeURIComponent(location.hash.slice(1));
    if (name && state.run?.name !== name) openRun(name).catch(toast);
    else if (!name) closeRun();
  } catch (error) { toast(error); }
});
async function boot() {
  api('/api/info').then(info => { state.maxUploadBytes = info.max_upload_bytes; }).catch(toast);
  probeEndpoint().catch(toast);
  await refreshSidebar();
  const wanted = decodeURIComponent(location.hash.slice(1));
  if (wanted && state.runs.some(run => run.name === wanted)) await openRun(wanted);
  else if (state.runs.length) await openRun(state.runs[0].name);
  else renderViewer();
}
boot().catch(toast);
