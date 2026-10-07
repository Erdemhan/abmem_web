/*
 * Demand-profile component (PLAN 2.8): editor of a synthetic demand profile and the demand preview chart.
 *
 * Markup: templates/components/demand_profile_form.html (editor) and
 * templates/components/demand_preview.html (chart and statistics table), both parameterised by an id prefix.
 * The server stays authoritative (abmem/services/market/demand_generator.py); validate() only mirrors its rules.
 *
 * API (global window.DemandProfileForm, no other globals are read):
 *   var form = DemandProfileForm.init(container, options)
 *       options.profile   initial profile or null (default: the include's JSON script, else null = file)
 *       options.defaults  profile used when the user first switches to "Generate"
 *       options.libraryId initial library profile id (PLAN 2.9, include with show_library=True), shown
 *                         read-only with source "From library"
 *       options.library   [{id, name}] library profiles offered by the "From library" select
 *       options.loadLibraryProfile(id, callback)  loads one library profile, callback(errorMessage, profile)
 *       The include flag generated_only=True (data-generated-only) keeps the source on "Generate".
 *   form.getSource()           'file' | 'generated' | 'library'
 *   form.getProfile()          null for the file, else the profile object sent to the server
 *   form.getLibraryId()        id of the selected library profile when the source is "From library", else null
 *   form.setProfile(profile, libraryId)  null selects the file, a libraryId selects "From library"
 *   form.setLibraryOptions(items)        replaces the options of the library select ([{id, name}])
 *   form.validate()            {field: message} keyed like the server ("seed", "day.mean", ...); {} when valid
 *   form.showErrors(errors)    shows server or client errors (keys with or without a "demand_profile." or
 *                              "profile." prefix, "demand_profile_source" for the library select)
 *   form.clearErrors()
 *   form.onChange(callback)    called after every change of the source or a profile field
 *   form.previewRequest(periods, extra)  body for the preview endpoint ({periods, source, profile?})
 *   DemandProfileForm.renderPreview(canvasEl, previewJson)  line chart with shaded night hours (Chart.js 3)
 *   DemandProfileForm.renderStats(tableBodyEl, previewJson) overall / day / night statistics rows
 *   DemandProfileForm.clearPreview(canvasEl, tableBodyEl)
 *   DemandProfileForm.createLoader(url, options)  debounced preview requests that ignore out-of-order replies:
 *       loader.request(body, callback)  callback(error, previewJson, body) for the latest request only
 *       loader.cancel()
 */
(function (global) {
    'use strict';

    var SEED_UPPER_BOUND = 2147483648;
    var DISTRIBUTIONS = ['NORMAL', 'UNIFORM', 'LOGNORMAL'];
    var SEGMENTS = ['day', 'night'];
    var SEGMENT_FIELDS = ['distribution', 'mean', 'std', 'min', 'max'];
    var DEFAULT_PROFILE = {
        version: 1,
        seed: 1,
        day_start_hour: 8,
        day_end_hour: 20,
        autocorrelation: 0.9,
        day: { distribution: 'NORMAL', mean: 42000, std: 4000, min: 30000, max: 55000 },
        night: { distribution: 'NORMAL', mean: 34000, std: 3000, min: 25000, max: 45000 }
    };

    function copy(value) { return JSON.parse(JSON.stringify(value)); }

    function usesMoments(distribution) { return distribution === 'NORMAL' || distribution === 'LOGNORMAL'; }

    function pad(hour) { return (hour < 10 ? '0' : '') + hour + ':00'; }

    // Number of a text field: a finite number, or the trimmed text itself (the server rejects it), or null.
    function numberOrText(text) {
        var trimmed = String(text === undefined || text === null ? '' : text).trim();
        if (trimmed === '') { return null; }
        var value = Number(trimmed);
        return isFinite(value) ? value : trimmed;
    }

    function isNumber(value) { return typeof value === 'number' && isFinite(value); }
    function isInteger(value) { return isNumber(value) && Math.floor(value) === value; }

    function nightText(start, end) {
        if (!isInteger(start) || !isInteger(end) || !(start >= 0 && end <= 24 && end - start >= 1 && end - start <= 23)) {
            return 'Night: -';
        }
        var hours = 24 - (end - start);
        return 'Night: ' + pad(end % 24) + '-' + pad(start === 0 ? 24 : start) + ' (' + hours + ' h)';
    }

    // Client-side mirror of DemandProfile.errors() (messages identical to the server).
    function validateProfile(profile) {
        var errors = {};
        if (!profile) { return errors; }
        if (!isInteger(profile.seed) || profile.seed < 0 || profile.seed >= SEED_UPPER_BOUND) {
            errors.seed = 'Demand seed must be an integer from 0 to ' + (SEED_UPPER_BOUND - 1) + '.';
        }
        var start = profile.day_start_hour;
        var end = profile.day_end_hour;
        if (!isInteger(start) || start < 0 || start > 23) {
            errors.day_start_hour = 'Day start hour must be from 0 to 23.';
        }
        if (!isInteger(end) || end < 1 || end > 24) {
            errors.day_end_hour = 'Day end hour must be from 1 to 24.';
        } else if (!errors.day_start_hour && !(end - start >= 1 && end - start <= 23)) {
            errors.day_end_hour = 'Day end hour must be after the day start hour, and day and night must each have '
                + 'at least one hour (day length 1 to 23 h).';
        }
        var phi = profile.autocorrelation;
        if (!isNumber(phi) || phi < 0 || phi >= 1) {
            errors.autocorrelation = 'Autocorrelation must be at least 0 and less than 1.';
        }
        SEGMENTS.forEach(function (name) {
            var segment = profile[name] || {};
            var label = name.charAt(0).toUpperCase() + name.slice(1) + ': ';
            if (DISTRIBUTIONS.indexOf(segment.distribution) === -1) {
                errors[name + '.distribution'] = label + 'the distribution must be one of ' + DISTRIBUTIONS.join(', ') + '.';
                return;
            }
            var needed = usesMoments(segment.distribution) ? ['mean', 'std', 'min', 'max'] : ['min', 'max'];
            var missing = false;
            needed.forEach(function (field) {
                if (segment[field] === null || segment[field] === undefined) {
                    errors[name + '.' + field] = label + field + ' is required for the ' + segment.distribution + ' distribution.';
                    missing = true;
                } else if (!isNumber(segment[field])) {
                    errors[name + '.' + field] = label + field + ' must be a number (MWh).';
                    missing = true;
                }
            });
            if (missing) { return; }
            if (!(segment.min > 0)) { errors[name + '.min'] = label + 'the minimum must be greater than 0 MWh.'; }
            if (!(segment.max > segment.min)) { errors[name + '.max'] = label + 'the maximum must be greater than the minimum.'; }
            if (usesMoments(segment.distribution)) {
                if (!(segment.std > 0)) { errors[name + '.std'] = label + 'the standard deviation must be greater than 0 MWh.'; }
                if (!(segment.mean > 0)) {
                    errors[name + '.mean'] = label + 'the mean must be greater than 0 MWh.';
                } else if (!(segment.min < segment.mean && segment.mean < segment.max)) {
                    errors[name + '.mean'] = label + 'the mean must lie strictly between the minimum and the maximum.';
                }
            }
        });
        return errors;
    }

    function init(container, options) {
        options = options || {};
        var prefix = container.getAttribute('data-prefix');
        var listeners = [];
        var defaults = copy(options.defaults || DEFAULT_PROFILE);
        var generatedOnly = container.getAttribute('data-generated-only') === 'true';
        var libraryRequest = 0;

        function field(id) { return document.getElementById(prefix + '-' + id); }
        function sourceRadios() { return container.querySelectorAll('input[name="' + prefix + '-source"]'); }
        function hasLibrary() { return Boolean(field('library-select')); }

        function getSource() {
            if (generatedOnly) { return 'generated'; }
            var checked = container.querySelector('input[name="' + prefix + '-source"]:checked');
            if (!checked) { return 'file'; }
            if (checked.value === 'library' && hasLibrary()) { return 'library'; }
            return checked.value === 'generated' ? 'generated' : 'file';
        }

        function setSource(source) {
            if (generatedOnly) { source = 'generated'; }
            Array.prototype.forEach.call(sourceRadios(), function (radio) { radio.checked = radio.value === source; });
            syncVisibility();
        }

        // In "From library" mode the profile fields show the library settings read-only.
        function setReadOnly(readOnly) {
            Array.prototype.forEach.call(field('generated').querySelectorAll('input, select, button'), function (node) {
                node.disabled = readOnly;
            });
        }

        function syncVisibility() {
            var source = getSource();
            field('generated').hidden = source === 'file';
            if (hasLibrary()) {
                field('library').hidden = source !== 'library';
                setReadOnly(source === 'library');
            }
            SEGMENTS.forEach(function (name) {
                var moments = usesMoments(field(name + '-distribution').value);
                field(name + '-mean-group').hidden = !moments;
                field(name + '-std-group').hidden = !moments;
            });
            field('night-text').textContent = nightText(
                numberOrText(field('day_start_hour').value), numberOrText(field('day_end_hour').value));
        }

        function readProfile() {
            var profile = {
                version: 1,
                seed: numberOrText(field('seed').value),
                day_start_hour: numberOrText(field('day_start_hour').value),
                day_end_hour: numberOrText(field('day_end_hour').value),
                autocorrelation: numberOrText(field('autocorrelation').value)
            };
            SEGMENTS.forEach(function (name) {
                var distribution = field(name + '-distribution').value;
                var segment = { distribution: distribution };
                (usesMoments(distribution) ? ['mean', 'std', 'min', 'max'] : ['min', 'max']).forEach(function (key) {
                    segment[key] = numberOrText(field(name + '-' + key).value);
                });
                profile[name] = segment;
            });
            return profile;
        }

        function writeProfile(profile) {
            field('seed').value = profile.seed;
            field('day_start_hour').value = profile.day_start_hour;
            field('day_end_hour').value = profile.day_end_hour;
            field('autocorrelation').value = profile.autocorrelation;
            SEGMENTS.forEach(function (name) {
                var segment = profile[name] || {};
                var fallback = defaults[name];
                field(name + '-distribution').value = DISTRIBUTIONS.indexOf(segment.distribution) !== -1
                    ? segment.distribution : fallback.distribution;
                ['mean', 'std', 'min', 'max'].forEach(function (key) {
                    var value = segment[key] !== undefined && segment[key] !== null ? segment[key] : fallback[key];
                    field(name + '-' + key).value = value;
                });
            });
        }

        function setLibraryOptions(items) {
            var select = field('library-select');
            if (!select) { return; }
            var current = select.value;
            while (select.options.length > 1) { select.remove(1); }
            (items || []).forEach(function (item) {
                var option = document.createElement('option');
                option.value = String(item.id);
                option.textContent = item.name;
                select.appendChild(option);
            });
            select.value = current;
            if (select.value !== current) { select.value = ''; }
        }

        function hasLibraryOption(id) {
            var select = field('library-select');
            return Boolean(select) && Array.prototype.some.call(select.options, function (option) {
                return option.value !== '' && option.value === String(id);
            });
        }

        function setProfile(profile, libraryId) {
            clearErrors();
            libraryRequest += 1;
            var linked = profile && libraryId !== undefined && libraryId !== null && hasLibraryOption(libraryId);
            if (hasLibrary()) { field('library-select').value = linked ? String(libraryId) : ''; }
            if (profile) {
                writeProfile(profile);
                setSource(linked ? 'library' : 'generated');
            } else {
                writeProfile(defaults);
                setSource(generatedOnly ? 'generated' : 'file');
            }
        }

        function getProfile() { return getSource() === 'file' ? null : readProfile(); }

        function getLibraryId() {
            if (getSource() !== 'library') { return null; }
            var value = field('library-select').value;
            return value === '' ? null : Number(value);
        }

        function validate() {
            var source = getSource();
            if (source === 'file') { return {}; }
            if (source === 'library' && getLibraryId() === null) {
                return { demand_profile_source: 'Select a demand profile of the library.' };
            }
            return validateProfile(readProfile());
        }

        function inputIdForKey(key) {
            if (key === 'demand_profile_source') { return hasLibrary() ? 'library' : 'source'; }
            var name = key.replace(/^(demand_profile|profile)\.?/, '');
            if (name === '' || name === 'version') { return 'source'; }
            return name.replace('.', '-');
        }

        function setError(id, message) {
            var input = field(id);
            var feedback = field(id + '-feedback');
            if (input && input.classList && input.tagName !== 'FIELDSET' && input.tagName !== 'DIV') {
                input.classList.toggle('is-invalid', Boolean(message));
            }
            if (feedback) {
                feedback.textContent = message || '';
                feedback.classList.toggle('d-block', Boolean(message));
            }
        }

        function clearErrors() {
            Array.prototype.forEach.call(container.querySelectorAll('.is-invalid'), function (node) {
                node.classList.remove('is-invalid');
            });
            Array.prototype.forEach.call(container.querySelectorAll('.invalid-feedback'), function (node) {
                node.textContent = '';
                node.classList.remove('d-block');
            });
        }

        // Shows errors keyed "seed", "day.mean" or "demand_profile.day.mean"; returns the number shown.
        function showErrors(errors) {
            clearErrors();
            var shown = 0;
            Object.keys(errors || {}).forEach(function (key) {
                var id = inputIdForKey(key);
                var message = [].concat(errors[key]).join(' ');
                if (!field(id + '-feedback')) { id = 'source'; }
                var previous = field(id + '-feedback').textContent;
                setError(id, previous ? previous + ' ' + message : message);
                shown += 1;
            });
            if (shown > 0 && getSource() === 'generated') { field('generated').hidden = false; }
            return shown;
        }

        function changed(event) {
            if (event && event.target && event.target.id) {
                setError(event.target.id.slice(prefix.length + 1), '');
            }
            syncVisibility();
            listeners.forEach(function (callback) { callback(); });
        }

        function notify() { listeners.forEach(function (callback) { callback(); }); }

        // Loads the selected library profile into the read-only fields (latest selection wins).
        function loadSelectedLibraryProfile() {
            var id = getLibraryId();
            var ticket = ++libraryRequest;
            setError('library', '');
            if (id === null || typeof options.loadLibraryProfile !== 'function') {
                notify();
                return;
            }
            options.loadLibraryProfile(id, function (message, profile) {
                if (ticket !== libraryRequest) { return; }
                if (message || !profile) {
                    setError('library', message || 'The demand profile could not be loaded.');
                } else {
                    writeProfile(profile);
                    syncVisibility();
                }
                notify();
            });
        }

        function previewRequest(periods, extra) {
            var body = { periods: periods };
            var profile = getProfile();
            if (profile) {
                body.source = 'generated';
                body.profile = profile;
            } else {
                body.source = 'file';
                // Shade the file preview with the day hours of the form.
                body.day_start_hour = numberOrText(field('day_start_hour').value);
                body.day_end_hour = numberOrText(field('day_end_hour').value);
            }
            Object.keys(extra || {}).forEach(function (key) { body[key] = extra[key]; });
            return body;
        }

        Array.prototype.forEach.call(sourceRadios(), function (radio) {
            radio.addEventListener('change', function (event) {
                if (radio.value === 'library' && radio.checked) {
                    syncVisibility();
                    loadSelectedLibraryProfile();
                    return;
                }
                changed(event);
            });
        });
        if (hasLibrary()) {
            setLibraryOptions(options.library || []);
            field('library-select').addEventListener('change', loadSelectedLibraryProfile);
            // "Edit copy": keep the settings, switch to a generated profile without the library link.
            field('library-edit').addEventListener('click', function () {
                field('library-select').value = '';
                setSource('generated');
                notify();
            });
        }
        ['input', 'change'].forEach(function (type) {
            field('generated').addEventListener(type, changed);
        });
        field('seed-random').addEventListener('click', function () {
            var buffer = new Uint32Array(1);
            global.crypto.getRandomValues(buffer);
            field('seed').value = String(buffer[0] % SEED_UPPER_BOUND);
            changed({ target: field('seed') });
        });

        var initial = options.profile;
        if (initial === undefined) {
            var script = document.getElementById(prefix + '-initial');
            initial = script ? JSON.parse(script.textContent) : null;
        }
        setProfile(initial || null, options.libraryId);

        return {
            getSource: getSource,
            getProfile: getProfile,
            getLibraryId: getLibraryId,
            setProfile: setProfile,
            setLibraryOptions: setLibraryOptions,
            validate: validate,
            showErrors: showErrors,
            clearErrors: clearErrors,
            onChange: function (callback) { listeners.push(callback); },
            previewRequest: previewRequest
        };
    }

    // ------------------------------------------------------------------ preview

    // Shades the night periods (preview.night_mask) behind the series.
    var nightShading = {
        id: 'demandNightShading',
        beforeDatasetsDraw: function (chart, args, options) {
            var mask = options && options.mask;
            if (!mask || !mask.length) { return; }
            var x = chart.scales.x;
            var area = chart.chartArea;
            var ctx = chart.ctx;
            ctx.save();
            ctx.fillStyle = 'rgba(58, 65, 111, 0.10)';
            var index = 0;
            while (index < mask.length) {
                if (!mask[index]) { index += 1; continue; }
                var first = index;
                while (index < mask.length && mask[index]) { index += 1; }
                var left = Math.max(area.left, x.getPixelForValue(first + 0.5));
                var right = Math.min(area.right, x.getPixelForValue(index + 0.5));
                if (right > left) { ctx.fillRect(left, area.top, right - left, area.bottom - area.top); }
            }
            ctx.restore();
        }
    };

    function formatMWh(value) {
        return Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 });
    }

    function clearPreview(canvasEl, tableBodyEl) {
        if (canvasEl && global.Chart && global.Chart.getChart) {
            var existing = global.Chart.getChart(canvasEl);
            if (existing) { existing.destroy(); }
        }
        if (tableBodyEl) { tableBodyEl.textContent = ''; }
    }

    function renderPreview(canvasEl, preview) {
        clearPreview(canvasEl, null);
        if (!global.Chart || !preview || !preview.values) { return null; }
        var points = preview.values.map(function (value, index) { return { x: index + 1, y: value }; });
        var stats = preview.stats || {};
        canvasEl.setAttribute('aria-label', 'Demand of periods 1 to ' + preview.periods + ' ('
            + (preview.source === 'generated' ? 'generated' : 'demand file') + '): min ' + formatMWh(stats.min)
            + ', mean ' + formatMWh(stats.mean) + ', max ' + formatMWh(stats.max) + ' MWh. Night hours are shaded.');
        return new global.Chart(canvasEl.getContext('2d'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Demand (MWh)', data: points, borderColor: '#cb0c9f', backgroundColor: '#cb0c9f',
                    pointRadius: points.length > 60 ? 0 : 2, borderWidth: 1.5, tension: 0
                }, {
                    label: 'Night hours', data: [], borderColor: 'rgba(58, 65, 111, 0.25)',
                    backgroundColor: 'rgba(58, 65, 111, 0.25)', pointStyle: 'rect'
                }]
            },
            options: {
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                interaction: { mode: 'nearest', axis: 'x', intersect: false },
                scales: {
                    x: { type: 'linear', min: 1, max: Math.max(preview.periods, 2),
                        title: { display: true, text: 'Period (hour)' }, ticks: { precision: 0 } },
                    y: { title: { display: true, text: 'Demand (MWh)' } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
                    demandNightShading: { mask: preview.night_mask }
                }
            },
            plugins: [nightShading]
        });
    }

    function renderStats(tableBodyEl, preview) {
        tableBodyEl.textContent = '';
        if (!preview || !preview.stats) { return; }
        var stats = preview.stats;
        [['Overall', stats], ['Day', stats.day], ['Night', stats.night]].forEach(function (pair) {
            var tr = document.createElement('tr');
            var th = document.createElement('th');
            th.scope = 'row';
            th.textContent = pair[0];
            tr.appendChild(th);
            var row = pair[1];
            ['count', 'min', 'mean', 'max', 'std'].forEach(function (key) {
                var td = document.createElement('td');
                td.className = 'text-end';
                td.textContent = row ? (key === 'count' ? String(row.count) : formatMWh(row[key])) : '-';
                tr.appendChild(td);
            });
            tableBodyEl.appendChild(tr);
        });
    }

    // Debounced preview requests; only the reply to the latest request reaches the callback.
    function createLoader(url, loaderOptions) {
        loaderOptions = loaderOptions || {};
        var delay = loaderOptions.delay === undefined ? 300 : loaderOptions.delay;
        var ticket = 0;
        var timer = null;

        function send(body, callback, current) {
            var headers = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
            if (loaderOptions.csrfToken) { headers['X-CSRFToken'] = loaderOptions.csrfToken(); }
            fetch(url, { method: 'POST', headers: headers, body: JSON.stringify(body) })
                .then(function (response) {
                    return response.json().catch(function () { return {}; }).then(function (json) {
                        return { ok: response.ok, status: response.status, body: json };
                    });
                })
                .then(function (result) {
                    if (current !== ticket) { return; }
                    if (result.ok) {
                        callback(null, result.body, body);
                    } else {
                        callback({ status: result.status, errors: result.body.errors || {} }, null, body);
                    }
                })
                .catch(function (error) {
                    if (current !== ticket) { return; }
                    callback({ status: 0, errors: { __all__: ['The demand preview could not be loaded: ' + error] } }, null, body);
                });
        }

        return {
            request: function (body, callback) {
                var current = ++ticket;
                clearTimeout(timer);
                timer = setTimeout(function () { send(body, callback, current); }, delay);
            },
            cancel: function () {
                ticket += 1;
                clearTimeout(timer);
            }
        };
    }

    global.DemandProfileForm = {
        init: init,
        validateProfile: validateProfile,
        nightText: nightText,
        renderPreview: renderPreview,
        renderStats: renderStats,
        clearPreview: clearPreview,
        createLoader: createLoader,
        DEFAULT_PROFILE: copy(DEFAULT_PROFILE)
    };
})(window);
