/*
 * script.js
 * ---------
 * Drives all five tools on the page.
 *
 * The dropdown contents (states, districts, crops, subdivisions, soil types)
 * are not hardcoded here — they are fetched once from /api/options, which
 * reads them straight out of the trained artifacts. That way the options a
 * user can pick always match what the models were actually trained on.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

let OPTIONS = null;

/* ------------------------------------------------------------------ *
 * Tabs
 * ------------------------------------------------------------------ */
$$('.tool-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    const tool = tab.dataset.tool;
    $$('.tool-tab').forEach((t) => t.classList.toggle('is-active', t === tab));
    $$('.tool-panel').forEach((p) =>
      p.classList.toggle('is-active', p.id === `panel-${tool}`)
    );
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});

/* ------------------------------------------------------------------ *
 * Dropdown helpers
 * ------------------------------------------------------------------ */
function fill(select, items, placeholder, labeller) {
  select.innerHTML = '';
  const first = document.createElement('option');
  first.value = '';
  first.textContent = placeholder;
  select.appendChild(first);

  (items || []).forEach((value) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = labeller ? labeller(value) : value;
    select.appendChild(opt);
  });

  select.disabled = !items || items.length === 0;
}

function disable(select, message) {
  select.innerHTML = `<option value="">${message}</option>`;
  select.disabled = true;
}

// District names in the source data are upper case with underscores.
// Show them as words without rewriting the values we send back.
const prettyPlace = (s) =>
  s.replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b[a-z]/g, (c) => c.toUpperCase());

const MONTH_NAMES = {
  JAN: 'January', FEB: 'February', MAR: 'March', APR: 'April',
  MAY: 'May', JUN: 'June', JUL: 'July', AUG: 'August',
  SEP: 'September', OCT: 'October', NOV: 'November', DEC: 'December',
  ANNUAL: 'Whole year',
  'Jan-Feb': 'Winter (Jan–Feb)',
  'Mar-May': 'Pre-monsoon (Mar–May)',
  'Jun-Sep': 'Monsoon (Jun–Sep)',
  'Oct-Dec': 'Post-monsoon (Oct–Dec)',
};

/* ------------------------------------------------------------------ *
 * Load the option lists once, then wire up the cascades
 * ------------------------------------------------------------------ */
async function loadOptions() {
  try {
    const res = await fetch('/api/options');
    OPTIONS = await res.json();
  } catch (err) {
    console.error('Could not load options', err);
    return;
  }

  const thisYear = new Date().getFullYear();

  /* --- 02 Crop prediction --- */
  const cp = OPTIONS.crop_prediction;
  const cpState = $('#cp-state');
  const cpDistrict = $('#cp-district');
  const cpSeason = $('#cp-season');

  if (cp) {
    fill(cpState, cp.states, 'Select a state');

    cpState.addEventListener('change', () => {
      const districts = cp.districts_by_state[cpState.value] || [];
      fill(cpDistrict, districts, 'Select a district', prettyPlace);
      disable(cpSeason, 'Select a district first');
    });

    cpDistrict.addEventListener('change', () => {
      const key = `${cpState.value}|${cpDistrict.value}`;
      const seasons = cp.seasons_by_district[key] || cp.seasons;
      fill(cpSeason, seasons, 'Select a season');
    });
  } else {
    disable(cpState, 'Unavailable on this server');
  }

  /* --- 03 Fertilizer --- */
  const fert = OPTIONS.fertilizer;
  if (fert) {
    fill($('#f-soil'), fert.soil_types, 'Select soil type');
    fill($('#f-crop'), fert.crop_types, 'Select crop type');
  } else {
    disable($('#f-soil'), 'Unavailable on this server');
    disable($('#f-crop'), 'Unavailable on this server');
  }

  /* --- 04 Rainfall --- */
  const rain = OPTIONS.rainfall;
  if (rain) {
    fill($('#r-subdivision'), rain.subdivisions, 'Select a subdivision', prettyPlace);
    fill($('#r-period'), rain.periods, 'Select a period', (p) => MONTH_NAMES[p] || p);
    $('#r-year').value = thisYear;
    $('#r-year').min = 1901;
  } else {
    disable($('#r-subdivision'), 'Unavailable on this server');
    disable($('#r-period'), 'Unavailable on this server');
  }

  /* --- 05 Yield --- */
  const yld = OPTIONS.yield;
  const yState = $('#y-state');
  const yDistrict = $('#y-district');
  const yCrop = $('#y-crop');

  if (yld) {
    fill(yState, yld.states, 'Select a state');
    fill($('#y-season'), yld.seasons, 'Select a season');
    $('#y-year').value = thisYear;

    yState.addEventListener('change', () => {
      fill(yDistrict, yld.districts_by_state[yState.value] || [], 'Select a district', prettyPlace);
      fill(yCrop, yld.crops_by_state[yState.value] || [], 'Select a crop');
    });
  } else {
    disable(yState, 'Unavailable on this server');
    disable(yDistrict, 'Unavailable on this server');
    disable(yCrop, 'Unavailable on this server');
  }
}

/* ------------------------------------------------------------------ *
 * Rendering helpers — every tool shows the same shape of result:
 * one headline figure, then a set of supporting details.
 * ------------------------------------------------------------------ */
const num = (n) =>
  Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });

function renderResult(area, { headline, headlineLabel, unit, chips, details, note }) {
  area.innerHTML = '';

  const main = document.createElement('div');
  main.className = 'result-main';
  main.innerHTML = `<span class="result-label">${headlineLabel}</span>`;

  const value = document.createElement('span');
  value.className = 'result-crop';
  value.textContent = headline;
  if (unit) {
    const u = document.createElement('span');
    u.className = 'result-unit';
    u.textContent = ` ${unit}`;
    value.appendChild(u);
  }
  main.appendChild(value);
  area.appendChild(main);

  if (chips && chips.items && chips.items.length) {
    const wrap = document.createElement('div');
    wrap.className = 'result-alts';
    wrap.innerHTML = `<span class="result-label">${chips.label}</span>`;
    const ul = document.createElement('ul');
    chips.items.forEach(({ text, meta }) => {
      const li = document.createElement('li');
      li.textContent = text;
      if (meta) {
        const span = document.createElement('span');
        span.textContent = meta;
        li.appendChild(span);
      }
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    area.appendChild(wrap);
  }

  if (details && details.length) {
    const dl = document.createElement('dl');
    dl.className = 'result-details';
    details.forEach(([term, def]) => {
      const dt = document.createElement('dt');
      dt.textContent = term;
      const dd = document.createElement('dd');
      dd.textContent = def;
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    area.appendChild(dl);
  }

  if (note) {
    const p = document.createElement('p');
    p.className = 'result-note';
    p.textContent = note;
    area.appendChild(p);
  }

  area.hidden = false;
}

/* ------------------------------------------------------------------ *
 * One submit handler for all five forms
 * ------------------------------------------------------------------ */
const ENDPOINTS = {
  recommendation: '/predict',
  prediction: '/api/crop-prediction',
  fertilizer: '/api/fertilizer',
  rainfall: '/api/rainfall',
  yield: '/api/yield',
};

const BUSY_TEXT = {
  recommendation: 'Checking the plot…',
  prediction: 'Reading the district records…',
  fertilizer: 'Matching the conditions…',
  rainfall: 'Fitting the rainfall trend…',
  yield: 'Working out the harvest…',
};

$$('form[data-form]').forEach((form) => {
  const tool = form.dataset.form;
  const panel = form.closest('.tool-panel');
  const resultArea = $('[data-result]', panel);
  const errorArea = $('[data-error]', panel);
  const button = $('.submit-btn', form);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    errorArea.hidden = true;
    resultArea.hidden = true;
    button.disabled = true;
    button.textContent = BUSY_TEXT[tool];

    const payload = {};
    new FormData(form).forEach((value, key) => {
      payload[key] = value;
    });

    try {
      const response = await fetch(ENDPOINTS[tool], {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Something went wrong reading the model.');
      }

      RENDERERS[tool](resultArea, data);
    } catch (err) {
      errorArea.textContent = err.message;
      errorArea.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = button.dataset.label;
    }
  });
});

/* ------------------------------------------------------------------ *
 * Per-tool result rendering
 * ------------------------------------------------------------------ */
const RENDERERS = {
  recommendation(area, data) {
    renderResult(area, {
      headlineLabel: 'Best match for this plot',
      headline: data.recommended_crop,
      chips: {
        label: 'Other close matches',
        items: data.top_3.map((i) => ({ text: i.crop, meta: `${i.confidence}%` })),
      },
    });
  },

  prediction(area, data) {
    renderResult(area, {
      headlineLabel: `Most commonly grown in ${prettyPlace(data.district)}, ${data.season}`,
      headline: data.recommended_crop,
      chips: {
        label: 'Share of the district\u2019s cropped area this season',
        items: data.crops.map((c) => ({ text: c.crop, meta: `${c.share}%` })),
      },
      note: data.crops[0].avg_area
        ? `${data.crops[0].crop} appears in ${data.crops[0].years} years of records here, on about ${num(data.crops[0].avg_area)} ha a year, producing around ${num(data.crops[0].avg_production)} units.`
        : null,
    });
  },

  fertilizer(area, data) {
    renderResult(area, {
      headlineLabel: 'Suggested fertilizer',
      headline: data.fertilizer,
      chips: data.top_3 && data.top_3.length > 1
        ? {
            label: 'Other candidates',
            items: data.top_3.map((i) => ({ text: i.fertilizer, meta: `${i.confidence}%` })),
          }
        : null,
      note: data.note,
    });
  },

  rainfall(area, data) {
    const [low, high] = data.typical_range_mm;
    const trend = data.trend_mm_per_decade;
    renderResult(area, {
      headlineLabel: `Expected rainfall · ${MONTH_NAMES[data.period] || data.period} ${data.year}`,
      headline: num(data.predicted_rainfall_mm),
      unit: 'mm',
      details: [
        ['Long-run average', `${num(data.historical_average_mm)} mm`],
        ['Usual range', `${num(low)} – ${num(high)} mm`],
        ['Recorded extremes', `${num(data.recorded_min_mm)} – ${num(data.recorded_max_mm)} mm`],
        ['Trend', `${trend >= 0 ? '+' : ''}${num(trend)} mm per decade`],
        ['Years of data', `${data.years_of_data}`],
      ],
      note: data.extrapolated
        ? 'This year falls outside the recorded period, so the figure is an extrapolation of the trend. Treat the usual range as the number to plan around.'
        : null,
    });
  },

  yield(area, data) {
    const details = [
      ['Yield per hectare', `${num(data.predicted_yield_per_unit_area)}`],
      ['Area planted', `${num(data.area)} ha`],
      ['Season', data.season],
    ];
    if (data.historical) {
      details.push([
        'Historical figure for this combination',
        `${num(data.historical.estimated_production)} (from ${data.historical.records} record${data.historical.records === 1 ? '' : 's'})`,
      ]);
    }
    renderResult(area, {
      headlineLabel: `Expected production · ${data.crop} in ${prettyPlace(data.district)}`,
      headline: num(data.predicted_production),
      details,
      note: data.historical
        ? null
        : 'No historical record for this exact combination, so the figure rests on the model alone.',
    });
  },
};

loadOptions();
