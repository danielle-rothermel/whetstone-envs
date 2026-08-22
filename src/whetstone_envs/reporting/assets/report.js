'use strict';

const payload = JSON.parse(document.getElementById('report-data').textContent);
const report = payload.report;
const state = {
  selected: [], trajectoryCandidateRef: null, resolution: null, task: null,
  step: null, bucket: null, filters: {}, wrap: true,
};
const byId = (value) => document.getElementById(value);

function node(tag, options = {}, children = []) {
  const value = document.createElement(tag);
  for (const [key, item] of Object.entries(options)) {
    if (key === 'className') value.className = item;
    else if (key === 'text') value.textContent = item;
    else if (key === 'dataset') {
      for (const [name, data] of Object.entries(item)) value.dataset[name] = data;
    } else if (key.startsWith('aria-')) value.setAttribute(key, item);
    else value[key] = item;
  }
  for (const child of children) value.append(child);
  return value;
}
function clear(value) { while (value.firstChild) value.firstChild.remove(); }
function button(text, action, className = '') {
  const value = node('button', { type: 'button', text, className });
  value.addEventListener('click', action);
  return value;
}
function panel(title) {
  const value = node('section', { className: 'panel' });
  value.append(node('h2', { text: title }));
  return value;
}
function textBlock(title, value) {
  const section = node('section');
  section.append(node('h3', { text: title }), button('Copy', () => copyExact(value), 'copy'), node('pre', { text: value, className: 'mono' }));
  return section;
}

function fallbackCopyExact(value) {
  const previous = document.activeElement;
  const textarea = node('textarea', { value, readOnly: true, tabIndex: -1, 'aria-hidden': 'true' });
  textarea.style.position = 'fixed';
  textarea.style.left = '-10000px';
  textarea.style.top = '0';
  textarea.style.opacity = '0';
  document.body.append(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, value.length);
  try {
    document.execCommand('copy');
    document.documentElement.dataset.copyFallback = 'complete';
    document.dispatchEvent(new CustomEvent('whetstone-copy-fallback', { detail: value }));
  } finally {
    textarea.remove();
    if (previous instanceof HTMLElement) previous.focus();
  }
}
function copyExact(value) {
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      const pending = navigator.clipboard.writeText(value);
      if (pending && typeof pending.catch === 'function') pending.catch(() => fallbackCopyExact(value));
      return;
    }
  } catch (_error) { fallbackCopyExact(value); return; }
  fallbackCopyExact(value);
}

function jsonText(value) { return value === null || value === undefined ? '—' : JSON.stringify(value, null, 2); }
function short(value) { return value.length > 12 ? value.slice(0, 12) : value; }
function preview(value) { return value.length > 180 ? `${value.slice(0, 180)}…` : value; }
function scoreText(value) { return value === null || value === undefined ? 'Incomplete — no numeric score' : `${(value * 100).toFixed(1)}%`; }
function signedScore(value) {
  if (value === null || value === undefined) return 'N/A';
  return `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)} points`;
}
function fragment() { return new URLSearchParams(location.hash.startsWith('#') ? location.hash.slice(1) : location.hash); }
function saveFragment() {
  const params = new URLSearchParams();
  if (payload.kind === 'trajectory') {
    if (state.trajectoryCandidateRef) params.append('candidate', state.trajectoryCandidateRef);
    if (state.step !== null) params.set('step', state.step);
    if (state.resolution !== null) params.set('resolution', state.resolution);
  } else for (const name of state.selected) params.append('candidate', name);
  if (state.task) params.set('task', state.task);
  if (state.bucket) params.set('bucket', state.bucket);
  for (const [key, value] of Object.entries(state.filters)) if (value) params.set(key, value);
  history.replaceState(null, '', `#${params.toString()}`);
}
function invalidFragment(message) {
  const notice = byId('fragment-notice');
  notice.hidden = false;
  notice.textContent = `Fragment notice: ${message}. No nearby item was selected.`;
}

function infoRows(title, rows) {
  const section = node('section');
  section.append(node('h3', { text: title }));
  const list = node('dl');
  for (const item of rows) list.append(node('dt', { text: item.name }), node('dd', { text: item.description }));
  section.append(list);
  return section;
}
function renderAbout() {
  const dialog = byId('about-dialog'); clear(dialog);
  dialog.append(
    node('h2', { id: 'about-title', text: 'About C19' }), node('p', { text: payload.info.objective }),
    infoRows('Public inputs and private gold', payload.info.public_inputs), infoRows('Grid sizes', payload.info.sizes),
    infoRows('Scenarios', payload.info.scenarios), infoRows('LRFPDT actions and no-op rules', payload.info.actions),
    infoRows('Grid tokens and coordinates', payload.info.tokens), infoRows('Facts and exact answer forms', payload.info.facts),
    node('h3', { text: 'Normalization and scoring' }), node('p', { text: payload.info.scoring }),
    node('h3', { text: 'Default pool and split roles' }), node('p', { text: payload.info.pool }),
    infoRows('Candidate terminology', payload.info.terminology), textBlock('Naive template', payload.info.naive_template),
    textBlock('Ceiling template', payload.info.ceiling_template), button('Close', () => dialog.close()),
  );
}
function renderHeader() {
  const header = byId('run-header'); clear(header);
  const run = payload.kind === 'eval' ? report.run : null;
  header.append(
    node('h1', { text: `C19 · ${payload.kind === 'eval' ? run.run_id : report.run_id}` }),
    node('span', { className: 'status', text: `Status: ${payload.kind === 'eval' ? report.results.map((item) => item.kind).join(', ') : report.terminal_status}` }),
    node('span', { text: payload.kind === 'eval' ? `${run.role} · ${report.tasks.length} tasks × ${run.repeats} repeats` : `${report.steps.length} steps · ${report.resolutions.length} resolutions` }),
    node('strong', { className: 'notice', text: 'PRIVATE ARTIFACT — contains gold, full outputs, prompts, and candidate text' }),
    button('About C19', () => byId('about-dialog').showModal()), button('Provenance', showProvenance),
  );
}
function showProvenance() {
  const dialog = byId('task-dialog'); clear(dialog);
  dialog.append(node('h2', { id: 'task-title', text: 'Exact provenance' }), textBlock('Report schema', report.schema_version));
  if (payload.kind === 'eval') for (const [key, value] of Object.entries(report.run)) dialog.append(textBlock(key, String(value)));
  else dialog.append(textBlock('Result ref', jsonText(report.result_ref)), textBlock('Mutation field', report.mutation_field));
  dialog.append(button('Close', () => dialog.close())); dialog.showModal();
}

function lineDiff(left, right) {
  const a = left.split('\n'); const b = right.split('\n'); const result = node('pre', { className: 'mono' });
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    if (a[index] === b[index]) result.append(node('span', { text: `  ${a[index] || ''}\n` }));
    else {
      if (a[index] !== undefined) result.append(node('span', { className: 'diff-del', text: `− ${a[index]}\n` }));
      if (b[index] !== undefined) result.append(node('span', { className: 'diff-add', text: `+ ${b[index]}\n` }));
    }
  }
  return result;
}
function candidateInspector(candidates, selectedIndex, textKey, compareIndex = null) {
  const section = panel('Candidate text and exact line diff'); section.classList.add('sticky');
  const candidate = candidates[selectedIndex]; if (!candidate) return section;
  section.append(
    node('p', { text: `${candidate.name || candidate.candidate_id} · ${candidate.source || candidate.dispositions.join(', ')} · ${short(candidate.identity_hash)}`, className: 'subtle' }),
    button('Copy exact raw value', () => copyExact(candidate[textKey])),
    button(state.wrap ? 'Use horizontal scroll' : 'Wrap long lines', () => { state.wrap = !state.wrap; render(); }),
  );
  const full = textBlock('Full exact text', candidate[textKey]); if (!state.wrap) full.classList.add('nowrap'); section.append(full);
  if (compareIndex !== null && candidates[compareIndex]) {
    const other = candidates[compareIndex];
    section.append(node('h3', { text: `Exact line diff against ${other.name || other.candidate_id}` }), lineDiff(other[textKey], candidate[textKey]));
  }
  return section;
}
function parseEvalSelection(names) {
  const requested = fragment().getAll('candidate');
  if (!requested.length) return names.slice(0, Math.min(2, names.length));
  if (requested.some((name) => !names.includes(name))) {
    invalidFragment(`unknown candidate coordinate ${requested.join(' / ')}`);
    return names.slice(0, Math.min(2, names.length));
  }
  return [...new Set(requested)].slice(0, 2);
}
function candidateRail(candidates) {
  const rail = panel('Candidates');
  for (const candidate of candidates) {
    const name = candidate.name; const active = state.selected.includes(name);
    const control = button(name, () => {
      state.selected = active ? state.selected.filter((item) => item !== name) : [...state.selected, name].slice(-2);
      if (!state.selected.length) state.selected = [name]; state.bucket = null; saveFragment(); render();
    });
    control.className = 'candidate'; control.setAttribute('aria-pressed', String(active));
    control.append(node('small', { text: `${candidate.source} · ${short(candidate.identity_hash)}` })); rail.append(control);
  }
  return rail;
}
function trajectoryCandidateRail() {
  const rail = panel('Candidates');
  for (const candidate of report.candidates) {
    const reference = candidate.record_ref.content_hash;
    const control = button(candidate.candidate_id, () => { state.trajectoryCandidateRef = reference; saveFragment(); render(); });
    control.className = 'candidate'; control.setAttribute('aria-pressed', String(state.trajectoryCandidateRef === reference));
    control.append(node('small', { text: `${candidate.dispositions.join(', ')} · ${short(candidate.identity_hash)}` })); rail.append(control);
  }
  return rail;
}

function resultFor(evalReport, name) { return evalReport.results.find((item) => item.candidate_name === name); }
function outcomeCards(evalReport, view, name) {
  const result = resultFor(evalReport, name); const section = panel(`Outcome accounting · ${name}`); const cards = node('div', { className: 'cards' });
  if (!result) cards.append(node('div', { className: 'card warn', text: 'No result' }));
  else if (result.kind !== 'success') cards.append(node('div', { className: 'card bad', text: `${result.kind}: ${result.classification}\n${result.message}` }));
  else {
    const accounting = result.accounting; const errors = view.provider_errors[name] || 0;
    const entries = [
      ['Score', `${scoreText(result.score)} · ${result.numerator}/${result.denominator} passed/planned`, result.score === null ? 'warn' : 'good'],
      ['Scored', String(accounting.present), 'good'], ['Failed', String(accounting.failed), accounting.failed ? 'bad' : 'good'],
      ['Missing', String(accounting.missing), accounting.missing ? 'missing' : 'good'], ['Invalid', String(accounting.invalid), accounting.invalid ? 'bad' : 'good'],
      ['Provider errors', String(errors), errors ? 'warn' : 'good'],
    ];
    for (const [title, value, status] of entries) cards.append(node('div', { className: `card ${status}` }, [node('strong', { text: title }), node('div', { text: value })]));
  }
  section.append(cards); return section;
}
function matrix(evalReport, view) {
  const section = panel('C19 scenario × size × fact matrix'); const grid = node('div', { className: 'matrix' });
  for (const size of ['small', 'medium']) for (const scenario of ['navigation', 'manipulation', 'door']) for (const fact of ['coordinate', 'heading', 'front', 'carrying']) {
    const label = `${scenario}|${size}|${fact}`; const summaries = state.selected.map((name) => view.matrices[name] && view.matrices[name][label]); const applicable = summaries.some(Boolean);
    const cell = button(`${size} · ${scenario} · ${fact}`, () => { if (applicable) { state.filters = { ...state.filters, scenario, size, fact }; saveFragment(); render(); } }, 'cell');
    if (!applicable) cell.append(node('div', { text: 'N/A', className: 'subtle' }));
    else for (let index = 0; index < summaries.length; index += 1) {
      const item = summaries[index]; cell.append(node('div', { text: item ? `${state.selected[index]}: ${item.numerator}/${item.denominator} · ${scoreText(item.score)} · F${item.accounting.failed} M${item.accounting.missing}` : `${state.selected[index]}: N/A` }));
    }
    grid.append(cell);
  }
  section.append(grid); return section;
}
function pairData(view) {
  if (state.selected.length !== 2) return null;
  return Object.values(view.pairs).find((item) => item.left === state.selected[0] && item.right === state.selected[1])
    || Object.values(view.pairs).find((item) => item.left === state.selected[1] && item.right === state.selected[0]) || null;
}
function pairedBuckets(view) {
  const section = panel('Paired observed outcomes'); const pair = pairData(view);
  if (!pair) { section.append(node('p', { text: 'Select exactly two successful candidates for paired accounting.' })); return section; }
  const labels = ['both correct', `${pair.left} only`, `${pair.right} only`, 'both wrong', 'execution mismatch'];
  const counts = Object.fromEntries(labels.map((item) => [item, 0])); for (const row of pair.rows) counts[row.bucket] = (counts[row.bucket] || 0) + 1;
  const buckets = node('div', { className: 'buckets' });
  for (const label of labels) buckets.append(button(`${label}\n${counts[label] || 0}`, () => { state.bucket = state.bucket === label ? null : label; saveFragment(); render(); }, `bucket ${state.bucket === label ? 'selected' : ''}`));
  section.append(buckets, node('p', { className: 'subtle', text: 'Execution mismatch is kept separate from semantic score disagreement. These are observed changes, not causal claims.' })); return section;
}
function filterControls() {
  const bar = node('div', { className: 'toolbar' });
  const definitions = [['scenario', ['', 'navigation', 'manipulation', 'door']], ['size', ['', 'small', 'medium']], ['fact', ['', 'coordinate', 'heading', 'front', 'carrying']], ['row_state', ['', 'scored', 'failed', 'missing', 'invalid']], ['score', ['', 'pass', 'wrong']], ['provider', ['', 'provider-error', 'no-provider-error']]];
  for (const [key, values] of definitions) {
    const select = node('select', { 'aria-label': key }); for (const value of values) select.append(node('option', { value, text: value || `all ${key.replace('_', ' ')}` }));
    select.value = state.filters[key] || ''; select.addEventListener('change', () => { state.filters[key] = select.value; saveFragment(); render(); }); bar.append(select);
  }
  const search = node('input', { type: 'search', placeholder: 'literal task ID', 'aria-label': 'literal task ID', value: state.filters.task_id || '' });
  search.addEventListener('change', () => { state.filters.task_id = search.value; saveFragment(); render(); });
  bar.append(search, button('Clear filters', () => { state.filters = {}; state.bucket = null; saveFragment(); render(); })); return bar;
}
function taskRows(evalReport, view) {
  const tasks = Object.fromEntries(evalReport.tasks.map((item) => [item.task_id, item])); const selected = new Set(state.selected); const pair = pairData(view);
  const buckets = new Map(pair ? pair.rows.map((item) => [`${item.task_id}:${item.seed_index}`, item.bucket]) : []); const order = { 'execution mismatch': 0 };
  if (pair) { order[`${pair.left} only`] = state.selected[0] === pair.left ? 2 : 1; order[`${pair.right} only`] = state.selected[0] === pair.right ? 2 : 1; }
  order['both wrong'] = 3; order['both correct'] = 4;
  return evalReport.observations.filter((row) => selected.has(row.candidate_name)).filter((row) => {
    const facet = view.task_facets[row.task_id]; const bucket = buckets.get(`${row.task_id}:${row.seed_index}`);
    return (!state.bucket || bucket === state.bucket) && (!state.filters.scenario || facet.scenario === state.filters.scenario)
      && (!state.filters.size || facet.size === state.filters.size) && (!state.filters.fact || facet.fact === state.filters.fact)
      && (!state.filters.row_state || row.state === state.filters.row_state) && (!state.filters.score || (state.filters.score === 'pass' ? row.score === 1 : row.score === 0))
      && (!state.filters.provider || (state.filters.provider === 'provider-error' ? row.provider_error !== null : row.provider_error === null))
      && (!state.filters.task_id || row.task_id.includes(state.filters.task_id));
  }).sort((a, b) => (order[buckets.get(`${a.task_id}:${a.seed_index}`)] ?? 5) - (order[buckets.get(`${b.task_id}:${b.seed_index}`)] ?? 5)
    || a.task_index - b.task_index || a.seed_index - b.seed_index || state.selected.indexOf(a.candidate_name) - state.selected.indexOf(b.candidate_name))
    .map((row) => ({ row, task: tasks[row.task_id], bucket: buckets.get(`${row.task_id}:${row.seed_index}`) || '' }));
}
function taskTable(evalReport, view) {
  const section = panel('Task observations'); const rows = taskRows(evalReport, view);
  section.append(filterControls(), node('p', { text: `Showing ${rows.length} observation rows. Complete run denominator remains ${evalReport.observations.length}.`, className: 'sr-status' }));
  const wrap = node('div', { className: 'table-wrap' }); const table = node('table'); const head = node('thead'); const headRow = node('tr');
  for (const label of ['Task', 'Strata / repeat', 'Candidate', 'Observed output', 'Gold', 'Score / state', 'Paired outcome', 'Detail']) headRow.append(node('th', { text: label }));
  head.append(headRow); const body = node('tbody');
  for (const item of rows) {
    const row = node('tr'); for (const value of [item.row.task_id, `${item.task.strata.join(', ')} · ${item.row.seed_index}`, item.row.candidate_name, preview(item.row.normalized_output ?? '—'), item.task.gold, rowState(item.row), item.bucket]) row.append(node('td', { text: value }));
    const detail = node('td'); detail.append(button('Open', () => openTask(evalReport, item.task, item.row.seed_index))); row.append(detail); body.append(row);
  }
  table.append(head, body); wrap.append(table); section.append(wrap); return section;
}
function rowState(row) { return row.state !== 'scored' ? `${row.state} · ${row.failure_code || 'no code'}` : (row.score === 1 ? '✓ correct' : '✕ wrong'); }
function openTask(evalReport, task, seedIndex) {
  state.task = `${task.task_id}:${seedIndex}`; saveFragment(); const dialog = byId('task-dialog'); clear(dialog);
  dialog.append(node('h2', { id: 'task-title', text: `${task.task_id} · repeat ${seedIndex}` }), node('p', { text: `hash ${task.task_hash} · seed ${task.seed} · strata ${task.strata.join(', ')}` }), textBlock('ASCII grid', task.prompt_inputs.grid), textBlock('Action script', task.prompt_inputs.command), textBlock('Question', task.prompt_inputs.question), textBlock('Private gold', task.gold));
  const rows = evalReport.observations.filter((item) => item.task_id === task.task_id && item.seed_index === seedIndex && state.selected.includes(item.candidate_name));
  const tabs = node('div', { className: 'tabs' }); const details = node('div', { className: 'detail-grid' });
  rows.forEach((item, index) => {
    tabs.append(button(item.candidate_name, () => { for (const child of details.children) child.classList.remove('active'); details.children[index].classList.add('active'); }));
    const candidate = evalReport.candidates.find((entry) => entry.name === item.candidate_name); const part = node('section', { className: index === 0 ? 'active' : '' });
    part.append(node('h3', { text: item.candidate_name }), textBlock('Full candidate template', candidate.prompt_template), textBlock('Rendered prompt', item.rendered_prompt), textBlock('Raw output', item.output_text ?? '—'), textBlock('Normalized output', item.normalized_output ?? '—'), textBlock('State, failure, and budgets', jsonText({ state: item.state, score: item.score, trace_state: item.trace_state, failure_code: item.failure_code, finish_reason: item.finish_reason, provider_error: item.provider_error, max_budget: item.max_budget, over_budget: item.over_budget })), textBlock('Submission result', jsonText(item.submission_result)), textBlock('Ordered component trace', jsonText(item.component_trace))); details.append(part);
  });
  dialog.append(tabs, details, button('Close', () => { state.task = null; saveFragment(); dialog.close(); })); dialog.showModal();
}
function evalSurface(evalReport, view, candidateOverride = null) {
  const candidates = evalReport.candidates; const names = candidates.map((item) => item.name);
  if (candidateOverride && names.includes(candidateOverride)) state.selected = [candidateOverride];
  if (!state.selected.length || state.selected.some((name) => !names.includes(name))) state.selected = parseEvalSelection(names);
  const selectedIndex = names.indexOf(state.selected[0]); const compareName = state.selected.find((name) => name !== state.selected[0]); const compareIndex = compareName === undefined ? null : names.indexOf(compareName);
  const layout = node('div', { className: 'layout' }); const side = node('aside'); const content = node('div');
  side.append(candidateRail(candidates), candidateInspector(candidates, selectedIndex, 'prompt_template', compareIndex));
  content.append(outcomeCards(evalReport, view, state.selected[0]), matrix(evalReport, view), pairedBuckets(view), taskTable(evalReport, view)); layout.append(side, content); return layout;
}
function renderEval() {
  byId('report').append(evalSurface(report, payload.view)); const rawTask = fragment().get('task');
  if (rawTask) { const [taskId, seedText] = rawTask.split(':'); const task = report.tasks.find((item) => item.task_id === taskId); const seed = Number(seedText); if (task && Number.isInteger(seed) && seed >= 0 && seed < report.run.repeats) openTask(report, task, seed); else invalidFragment(`unknown task coordinate ${rawTask}`); }
}

function candidateForRef(reference) { return report.candidates.find((item) => item.record_ref.content_hash === reference); }
function resolutionKey(resolution) { return `${resolution.step_index}:${resolution.resolution_index}`; }
function resolutionForKey(key) { return report.resolutions.find((item) => resolutionKey(item) === key); }
function restoreTrajectoryCoordinates() {
  const params = fragment(); const requested = params.getAll('candidate')[0]; const known = new Set(report.candidates.map((item) => item.record_ref.content_hash));
  if (requested) { if (known.has(requested)) state.trajectoryCandidateRef = requested; else invalidFragment(`unknown candidate coordinate ${requested}`); }
  if (!state.trajectoryCandidateRef) state.trajectoryCandidateRef = report.candidates[0].record_ref.content_hash;
  const rawStep = params.get('step');
  if (rawStep !== null) { if (report.steps.some((item) => String(item.step_index) === rawStep)) state.step = rawStep; else invalidFragment(`unknown step coordinate ${rawStep}`); }
  const rawResolution = params.get('resolution');
  if (rawResolution !== null) {
    const selected = resolutionForKey(rawResolution);
    if (!selected) invalidFragment(`unknown resolution coordinate ${rawResolution}`);
    else if (state.step !== null && state.step !== String(selected.step_index)) invalidFragment(`resolution ${rawResolution} is not in step ${state.step}`);
    else { state.resolution = rawResolution; state.step = String(selected.step_index); }
  } else if (state.step !== null) {
    const first = report.resolutions.find((item) => String(item.step_index) === state.step); if (first) state.resolution = resolutionKey(first);
  }
  if (state.resolution === null && report.resolutions.length) { state.resolution = resolutionKey(report.resolutions[0]); state.step = String(report.resolutions[0].step_index); }
}
function trajectorySurfaces() {
  const outer = node('div'); const timeline = panel('Ordered step and exact resolution timeline');
  for (const step of report.steps) {
    const stepSection = node('section', { className: 'timeline-step' });
    const terminalStep = step.accepted_candidates.some((reference) => report.terminal_candidate_refs.some((terminal) => terminal.content_hash === reference.content_hash));
    stepSection.append(node('h3', { text: `Step ${step.step_index} · ${step.status}` }), node('p', { text: `${step.proposed_candidates.length} proposed · ${step.accepted_candidates.length} accepted · terminal selection ${terminalStep ? 'yes' : 'no'}` }), node('pre', { className: 'mono', text: `step delta ${jsonText(step.budget_delta_consumed)}\ncumulative ${jsonText(step.budget_cumulative_consumed)}\nremaining ${jsonText(step.budget_remaining)}\nstep failure ${jsonText(step.terminal_failure)}` }));
    const resolutions = report.resolutions.filter((item) => item.step_index === step.step_index); const points = node('div', { className: 'timeline' });
    for (const resolution of resolutions) {
      const key = resolutionKey(resolution); const candidate = candidateForRef(resolution.candidate_ref.content_hash); const terminal = report.terminal_candidate_refs.some((item) => item.content_hash === resolution.candidate_ref.content_hash);
      const control = button(`Resolution ${resolution.resolution_index} · ${resolution.outcome} · ${resolution.classification}\n${candidate.candidate_id}\nreward ${jsonText(resolution.reward)}\nfailure ${jsonText(resolution.terminal_failure)}\nterminal selection ${terminal ? 'yes' : 'no'}`, () => { state.step = String(resolution.step_index); state.resolution = key; state.trajectoryCandidateRef = resolution.candidate_ref.content_hash; saveFragment(); render(); });
      control.dataset.resolution = key; if (state.resolution === key) control.classList.add('selected'); points.append(control);
    }
    if (!resolutions.length) points.append(node('p', { text: 'No evaluation resolutions.' })); stepSection.append(points); timeline.append(stepSection);
  }
  const lineage = panel('Exact candidate lineage'); const lineageGrid = node('div', { className: 'lineage' }); const known = new Set(report.candidates.map((item) => item.record_ref.content_hash)); const external = [];
  for (const candidate of report.candidates) if (!known.has(candidate.base_ref.content_hash) && !external.includes(candidate.base_ref.content_hash)) external.push(candidate.base_ref.content_hash);
  for (const root of external) lineageGrid.append(node('button', { type: 'button', className: 'external', text: `External root\n${short(root)}`, disabled: true }));
  for (const candidate of report.candidates) {
    const reference = candidate.record_ref.content_hash; const terminal = report.terminal_candidate_refs.some((item) => item.content_hash === reference); const parent = candidate.base_candidate_ref ? short(candidate.base_candidate_ref.content_hash) : `external ${short(candidate.base_ref.content_hash)}`;
    const control = button(`${candidate.candidate_id}${terminal ? ' · terminal' : ''}\n${candidate.dispositions.join(', ')}\nparent ${parent}`, () => { state.trajectoryCandidateRef = reference; saveFragment(); render(); });
    if (state.trajectoryCandidateRef === reference) control.classList.add('selected'); lineageGrid.append(control);
  }
  lineage.append(lineageGrid); outer.append(timeline, lineage);
  const spend = spendPanel(); if (spend) outer.append(spend);
  return outer;
}
function spendUsd(role) {
  // A total appears only when every billable call carried a price.
  return role.usd === null || role.usd === undefined ? `unpriced (${role.unpriced_calls}/${role.calls})` : `$${role.usd.toFixed(6)}`;
}
function spendPanel() {
  if (!report.spend) return null;
  const surface = panel('Run spend'); const grid = node('div', { className: 'lineage' });
  for (const role of [report.spend.task_model, report.spend.proposer]) {
    grid.append(node('pre', { className: 'mono', text: `${role.role}\ncalls ${role.calls} (cached ${role.cached_calls})\ntokens in ${role.input_tokens} / out ${role.output_tokens}\npriced ${role.priced_calls} · unpriced ${role.unpriced_calls}\nrows missing token breakdown ${role.rows_missing_token_breakdown}\nusd ${spendUsd(role)}` }));
  }
  surface.append(grid); return surface;
}
function trajectoryDiagnosis(resolution, candidate) {
  const key = resolutionKey(resolution); const diagnosis = payload.view.diagnoses[key]; const step = report.steps[resolution.step_index]; const section = panel(`Observed change diagnosis · step ${key}`);
  section.append(node('p', { text: 'This surface reports the exact then-current observed comparison alongside the prompt mutation and does not imply causation.' }), node('div', { className: 'cards' }, [node('div', { className: 'card', text: `Fail-to-pass\n${resolution.gains === null ? 'N/A' : resolution.gains}` }), node('div', { className: 'card', text: `Pass-to-fail\n${resolution.regressions === null ? 'N/A' : resolution.regressions}` }), node('div', { className: 'card', text: `Execution mismatches\n${resolution.execution_mismatches === null ? 'N/A' : resolution.execution_mismatches}` }), node('div', { className: 'card', text: `Change budget\n${jsonText(step.budget_delta_consumed)}` })]));
  if (diagnosis && diagnosis.overall) {
    const overall = diagnosis.overall; section.append(node('h3', { text: 'Overall observed change' }), node('p', { text: `${scoreText(overall.before)} (${overall.before_numerator}/${overall.denominator}) → ${scoreText(overall.after)} (${overall.after_numerator}/${overall.denominator}); Δ ${signedScore(overall.change)}` }));
  }
  if (diagnosis && diagnosis.strata.length) {
    const table = node('table'); const head = node('thead'); const headRow = node('tr'); for (const label of ['Stratum', 'Before', 'After', 'Observed change']) headRow.append(node('th', { text: label })); head.append(headRow); const body = node('tbody');
    for (const item of diagnosis.strata) { const row = node('tr'); for (const value of [item.stratum, `${scoreText(item.before)} · ${item.before_numerator}/${item.denominator}`, `${scoreText(item.after)} · ${item.after_numerator}/${item.denominator}`, signedScore(item.change)]) row.append(node('td', { text: value })); body.append(row); }
    table.append(head, body); section.append(node('h3', { text: 'Per-stratum observed changes' }), table);
  }
  if (diagnosis && diagnosis.changed_rows.length) {
    const links = node('div', { className: 'buckets' });
    for (const changed of diagnosis.changed_rows) links.append(button(`${changed.bucket.replaceAll('_', '-')} · ${changed.task_id} · repeat ${changed.seed_index}`, () => { const evalReport = resolution.eval_report; const task = evalReport.tasks.find((item) => item.task_id === changed.task_id); state.selected = [evalReport.candidates[0].name]; openTask(evalReport, task, changed.seed_index); }));
    section.append(node('h3', { text: 'Exact changed observation rows' }), links);
  } else section.append(node('p', { className: 'subtle', text: diagnosis ? 'No changed observation coordinates were recorded.' : 'No exact then-current parent evaluation snapshot is available for this resolution.' }));
  const parent = candidate.base_candidate_ref && candidateForRef(candidate.base_candidate_ref.content_hash);
  if (parent) section.append(node('h3', { text: `Full prompt diff against ${parent.candidate_id}` }), lineDiff(parent.mutation_text, candidate.mutation_text));
  else section.append(node('p', { text: 'Base candidate is an explicit external root; no parent text is available.' }));
  return section;
}
function renderTrajectory() {
  restoreTrajectoryCoordinates(); const candidate = candidateForRef(state.trajectoryCandidateRef) || report.candidates[0]; const resolution = state.resolution === null ? null : resolutionForKey(state.resolution); const root = byId('report'); root.append(trajectorySurfaces());
  const layout = node('div', { className: 'layout' }); const side = node('aside'); const content = node('div'); side.append(trajectoryCandidateRail(), candidateInspector(report.candidates, report.candidates.indexOf(candidate), 'mutation_text'));
  if (resolution) {
    const resolutionCandidate = candidateForRef(resolution.candidate_ref.content_hash); content.append(trajectoryDiagnosis(resolution, resolutionCandidate));
    if (resolution.eval_report) { const key = resolutionKey(resolution); const view = payload.view.eval_views[key]; state.selected = [resolution.eval_report.candidates[0].name]; content.append(node('h2', { text: `Hydrated evaluation · step ${resolution.step_index}, resolution ${resolution.resolution_index}` }), outcomeCards(resolution.eval_report, view, state.selected[0]), matrix(resolution.eval_report, view), taskTable(resolution.eval_report, view)); }
    else content.append(node('p', { className: 'panel', text: 'No hydrated evaluation is recorded for this exact resolution.' }));
  } else content.append(node('p', { className: 'panel', text: 'No evaluation resolutions are recorded.' }));
  layout.append(side, content); root.append(layout);
}
function restoreFilters() {
  const params = fragment(); for (const key of ['scenario', 'size', 'fact', 'row_state', 'score', 'provider', 'task_id']) if (params.has(key)) state.filters[key] = params.get(key); state.bucket = params.get('bucket');
}
function render() { clear(byId('report')); if (payload.kind === 'eval') renderEval(); else renderTrajectory(); }

renderAbout(); renderHeader(); restoreFilters(); render();
