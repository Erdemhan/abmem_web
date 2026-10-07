/*
 * Demand-profile component (PLAN 2.8): editor of a synthetic demand profile and the demand preview chart.
 *
 * Markup: templates/components/demand_profile_form.html (editor, with components/demand_shape_fields.html
 * included once for the whole run or summer and once with the prefix "<prefix>-winter" for winter) and
 * templates/components/demand_preview.html (chart and statistics table), both parameterised by an id prefix.
 * The server stays authoritative (abmem/services/market/demand_generator.py). validateProfile() only mirrors
 * its rules and messages.
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
 *                              (seasons: null while the summer/winter split is off)
 *   form.getLibraryId()        id of the selected library profile when the source is "From library", else null
 *   form.setProfile(profile, libraryId)  null selects the file, a libraryId selects "From library"
 *   form.setLibraryOptions(items)        replaces the options of the library select ([{id, name}])
 *   form.setHorizon(periods, warnNoun)   number of periods shown in the season text ("Winter: periods S+1..N").
 *                              With warnNoun (for example 'run') a warning "The whole run is summer." is shown
 *                              when the last summer period is at or after the horizon.
 *   form.validate()            {field: message} keyed like the server ("seed", "day.mean",
 *                              "seasons.winter.night.std", ...); {} when valid
 *   form.showErrors(errors)    shows server or client errors (keys with or without a "demand_profile." or
 *                              "profile." prefix, "demand_profile_source" for the library select)
 *   form.clearErrors()
 *   form.onChange(callback)    called after every change of the source or a profile field
 *   form.previewRequest(periods, extra)  body for the preview endpoint ({periods, source, profile?})
 *   DemandProfileForm.renderPreview(canvasEl, previewJson)  line chart with shaded night hours and a dashed
 *                              line at the season boundary (Chart.js 3)
 *   DemandProfileForm.renderStats(tableBodyEl, previewJson) overall / day / night (/ summer / winter) rows
 *   DemandProfileForm.clearPreview(canvasEl, tableBodyEl)
 *   DemandProfileForm.createLoader(url, options)  debounced preview requests that ignore out-of-order replies:
 *       loader.request(body, callback)  callback(error, previewJson, body) for the latest request only
 *       loader.cancel()
 */
(function (global) {
    'use strict';

    var SEED_UPPER_BOUND = 2147483648;
    var MAX_RAMP_HOURS = 12;
    var MAX_SEASON_TRANSITION_DAYS = 90;
    var MAX_SUMMER_END_PERIOD = 1000000;
    var HOURS_PER_DAY = 24;
    var DISTRIBUTIONS = ['NORMAL', 'UNIFORM', 'LOGNORMAL'];
    var SEGMENTS = ['day', 'night'];
    var SEGMENT_KEYS = ['distribution', 'mean', 'std', 'min', 'max'];
    var SEASON_KEYS = ['summer_end_period', 'transition_days', 'winter'];
    var RAMPS = [['morning_ramp_hours', 'Morning transition'], ['evening_ramp_hours', 'Evening transition']];
    var DEFAULT_PROFILE = {
        version: 1,
        seed: 1,
        day_start_hour: 8,
        day_end_hour: 20,
        morning_ramp_hours: 0,
        evening_ramp_hours: 0,
        autocorrelation: 0.9,
        day: { distribution: 'NORMAL', mean: 42000, std: 4000, min: 30000, max: 55000 },
        night: { distribution: 'NORMAL', mean: 34000, std: 3000, min: 25000, max: 45000 },
        seasons: null
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

    // Like numberOrText, but an empty field is 0 (the server default of the transition widths).
    function numberOrZero(text) {
        var value = numberOrText(text);
        return value === null ? 0 : value;
    }

    function isNumber(value) { return typeof value === 'number' && isFinite(value); }
    function isInteger(value) { return isNumber(value) && Math.floor(value) === value; }
    function isMapping(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }

    function nightText(start, end) {
        if (!isInteger(start) || !isInteger(end) || !(start >= 0 && end <= 24 && end - start >= 1 && end - start <= 23)) {
            return 'Night: -';
        }
        var hours = 24 - (end - start);
        return 'Night: ' + pad(end % 24) + '-' + pad(start === 0 ? 24 : start) + ' (' + hours + ' h)';
    }

    // "Summer: periods 1..S, Winter: periods S+1..N" (N = horizon when known).
    function seasonText(summerEnd, horizon) {
        if (!isInteger(summerEnd) || summerEnd < 1) { return 'Summer: -'; }
        var known = isInteger(horizon) && horizon >= 1;
        if (known && summerEnd >= horizon) {
            return 'Summer: periods 1..' + horizon + ', Winter: none';
        }
        return 'Summer: periods 1..' + summerEnd + ', Winter: periods ' + (summerEnd + 1) + '..'
            + (known ? horizon : '');
    }

    // ------------------------------------------------------------------ validation (mirror of the server)

    // Mirrors demand_generator._integer: an integral number (or digit string), else null.
    function toInteger(raw) {
        if (typeof raw === 'number') { return isInteger(raw) ? raw : null; }
        if (typeof raw === 'string' && /^-?\d+$/.test(raw.trim())) { return Number(raw.trim()); }
        return null;
    }

    // Mirrors demand_generator._number: a finite number (or numeric string), else null.
    function toNumber(raw) {
        if (typeof raw === 'number') { return isFinite(raw) ? raw : null; }
        if (typeof raw === 'string' && raw.trim() !== '') {
            var value = Number(raw.trim());
            return isFinite(value) ? value : null;
        }
        return null;
    }

    function segmentLabel(key) {
        var parts = key.split('.');
        var last = parts[parts.length - 1];
        if (parts.slice(0, -1).indexOf('winter') !== -1) { return 'Winter ' + last; }
        return last.charAt(0).toUpperCase() + last.slice(1);
    }

    function addError(errors, key, message) {
        errors[key] = errors[key] ? errors[key] + ' ' + message : message;
    }

    // Mirrors _parse_segment: returns the segment or null, parse errors go to errors.
    function parseSegment(raw, key, errors) {
        var label = segmentLabel(key) + ': ';
        if (!isMapping(raw)) {
            addError(errors, key, label + 'the segment must be an object with distribution, min and max.');
            return null;
        }
        var unknown = Object.keys(raw).filter(function (name) { return SEGMENT_KEYS.indexOf(name) === -1; }).sort();
        if (unknown.length) { addError(errors, key, label + 'unknown field(s) ' + unknown.join(', ') + '.'); }
        var distribution = typeof raw.distribution === 'string' ? raw.distribution.trim().toUpperCase() : raw.distribution;
        if (DISTRIBUTIONS.indexOf(distribution) === -1) {
            addError(errors, key + '.distribution', label + 'the distribution must be one of ' + DISTRIBUTIONS.join(', ') + '.');
            return null;
        }
        var needed = usesMoments(distribution) ? ['mean', 'std', 'min', 'max'] : ['min', 'max'];
        var segment = { distribution: distribution };
        var ok = true;
        needed.forEach(function (field) {
            var value = raw[field];
            if (value === null || value === undefined || value === '') {
                addError(errors, key + '.' + field, label + field + ' is required for the ' + distribution + ' distribution.');
                ok = false;
                return;
            }
            var number = toNumber(value);
            if (number === null) {
                addError(errors, key + '.' + field, label + field + ' must be a number (MWh).');
                ok = false;
                return;
            }
            segment[field] = number;
        });
        return ok ? segment : null;
    }

    // Mirrors Segment.errors.
    function segmentErrors(segment, key, errors) {
        var label = segmentLabel(key) + ': ';
        if (!(segment.min > 0)) { addError(errors, key + '.min', label + 'the minimum must be greater than 0 MWh.'); }
        if (!(segment.max > segment.min)) { addError(errors, key + '.max', label + 'the maximum must be greater than the minimum.'); }
        if (usesMoments(segment.distribution)) {
            if (!(segment.std > 0)) { addError(errors, key + '.std', label + 'the standard deviation must be greater than 0 MWh.'); }
            if (!(segment.mean > 0)) {
                addError(errors, key + '.mean', label + 'the mean must be greater than 0 MWh.');
            } else if (!(segment.min < segment.mean && segment.mean < segment.max)) {
                addError(errors, key + '.mean', label + 'the mean must lie strictly between the minimum and the maximum.');
            }
        }
    }

    // Mirrors DailyShape.parse (prefix '' or 'seasons.winter.').
    function parseShape(data, prefix, errors) {
        var shape = {};
        var ok = true;
        [['day_start_hour', 'Day start hour'], ['day_end_hour', 'Day end hour']].forEach(function (pair) {
            var value = toInteger(data[pair[0]]);
            if (value === null) {
                errors[prefix + pair[0]] = pair[1] + ' is required and must be an integer.';
                ok = false;
            } else {
                shape[pair[0]] = value;
            }
        });
        RAMPS.forEach(function (pair) {
            var raw = data[pair[0]];
            var value = raw === undefined || raw === null || raw === '' ? 0 : toInteger(raw);
            if (value === null) {
                errors[prefix + pair[0]] = pair[1] + ' must be a whole number of hours.';
                ok = false;
            } else {
                shape[pair[0]] = value;
            }
        });
        SEGMENTS.forEach(function (name) { shape[name] = parseSegment(data[name], prefix + name, errors); });
        if (!ok || shape.day === null || shape.night === null) {
            SEGMENTS.forEach(function (name) {
                if (shape[name] !== null) { segmentErrors(shape[name], prefix + name, errors); }
            });
            return null;
        }
        return shape;
    }

    // Mirrors DailyShape.errors.
    function shapeErrors(shape, prefix, errors) {
        var own = {};
        var start = shape.day_start_hour;
        var end = shape.day_end_hour;
        if (!(start >= 0 && start <= 23)) { own[prefix + 'day_start_hour'] = 'Day start hour must be from 0 to 23.'; }
        if (!(end >= 1 && end <= 24)) {
            own[prefix + 'day_end_hour'] = 'Day end hour must be from 1 to 24.';
        } else if (!own[prefix + 'day_start_hour'] && !(end - start >= 1 && end - start <= 23)) {
            own[prefix + 'day_end_hour'] = 'Day end hour must be after the day start hour, and day and night must each '
                + 'have at least one hour (day length 1 to 23 h).';
        }
        RAMPS.forEach(function (pair) {
            var width = shape[pair[0]];
            if (!(width >= 0 && width <= MAX_RAMP_HOURS)) {
                own[prefix + pair[0]] = pair[1] + ' must be from 0 to ' + MAX_RAMP_HOURS + ' hours.';
            }
        });
        if (Object.keys(own).length === 0) {
            var dayHours = end - start;
            var limit = 2 * Math.min(dayHours, HOURS_PER_DAY - dayHours);
            if (shape.morning_ramp_hours + shape.evening_ramp_hours > limit) {
                own[prefix + 'evening_ramp_hours'] = 'The two transitions overlap. Morning plus evening transition '
                    + 'must be at most ' + limit + ' hours (twice the shorter of day and night).';
            }
        }
        Object.keys(own).forEach(function (key) { errors[key] = own[key]; });
        SEGMENTS.forEach(function (name) { segmentErrors(shape[name], prefix + name, errors); });
    }

    // Mirrors Seasons.parse.
    function parseSeasons(data, errors) {
        var prefix = 'seasons.';
        if (!isMapping(data)) {
            errors.seasons = 'Seasons must be an object or null.';
            return null;
        }
        var unknown = Object.keys(data).filter(function (name) { return SEASON_KEYS.indexOf(name) === -1; }).sort();
        if (unknown.length) { errors.seasons = 'Unknown seasons field(s): ' + unknown.join(', ') + '.'; }
        var seasons = {};
        var ok = true;
        var raw = data.summer_end_period;
        var summerEnd = toInteger(raw === undefined ? null : raw);
        if (summerEnd === null) {
            errors[prefix + 'summer_end_period'] = 'Summer end period is required and must be a whole number.';
            ok = false;
        } else {
            seasons.summer_end_period = summerEnd;
        }
        raw = data.transition_days;
        var days = raw === undefined || raw === null || raw === '' ? 0 : toInteger(raw);
        if (days === null) {
            errors[prefix + 'transition_days'] = 'Season transition is required and must be a whole number.';
            ok = false;
        } else {
            seasons.transition_days = days;
        }
        if (!isMapping(data.winter)) {
            errors[prefix + 'winter'] = 'Winter settings are required.';
            return null;
        }
        seasons.winter = parseShape(data.winter, prefix + 'winter.', errors);
        return seasons.winter === null || !ok ? null : seasons;
    }

    // Mirrors Seasons.errors.
    function seasonsErrors(seasons, errors) {
        var prefix = 'seasons.';
        var summerEnd = seasons.summer_end_period;
        var days = seasons.transition_days;
        var endBad = !(summerEnd >= 1 && summerEnd <= MAX_SUMMER_END_PERIOD);
        if (endBad) {
            errors[prefix + 'summer_end_period'] = 'Summer end period must be from 1 to ' + MAX_SUMMER_END_PERIOD + '.';
        }
        if (!(days >= 0 && days <= MAX_SEASON_TRANSITION_DAYS)) {
            errors[prefix + 'transition_days'] = 'Season transition must be from 0 to ' + MAX_SEASON_TRANSITION_DAYS + ' days.';
        } else if (!endBad && HOURS_PER_DAY * days > 2 * summerEnd) {
            errors[prefix + 'transition_days'] = 'The season transition starts before period 1. Half of the '
                + 'transition (in hours) must not exceed the summer end period.';
        }
        shapeErrors(seasons.winter, prefix + 'winter.', errors);
    }

    // Client-side mirror of DemandProfile.from_dict (messages identical to the server).
    function validateProfile(profile) {
        var errors = {};
        if (!profile) { return errors; }
        var seed = toInteger(profile.seed);
        if (seed === null) { errors.seed = 'Demand seed is required and must be an integer.'; }
        var phi = toNumber(profile.autocorrelation);
        if (phi === null) { errors.autocorrelation = 'Autocorrelation is required and must be a number.'; }
        var summer = parseShape(profile, '', errors);
        var seasonsRaw = profile.seasons === undefined ? null : profile.seasons;
        var seasons = seasonsRaw === null ? null : parseSeasons(seasonsRaw, errors);
        if (Object.keys(errors).length > 0 || seed === null || phi === null || summer === null
                || (seasonsRaw !== null && seasons === null)) {
            // Like the server, report the semantic errors of the parts that did parse as well.
            if (summer !== null) { shapeErrors(summer, '', errors); }
            if (seasons !== null) { seasonsErrors(seasons, errors); }
            return errors;
        }
        if (!(seed >= 0 && seed < SEED_UPPER_BOUND)) {
            errors.seed = 'Demand seed must be an integer from 0 to ' + (SEED_UPPER_BOUND - 1) + '.';
        }
        if (!(phi >= 0 && phi < 1)) { errors.autocorrelation = 'Autocorrelation must be at least 0 and less than 1.'; }
        shapeErrors(summer, '', errors);
        if (seasons !== null) { seasonsErrors(seasons, errors); }
        return errors;
    }

    // ------------------------------------------------------------------ editor

    function init(container, options) {
        options = options || {};
        var prefix = container.getAttribute('data-prefix');
        var listeners = [];
        var defaults = copy(options.defaults || DEFAULT_PROFILE);
        var generatedOnly = container.getAttribute('data-generated-only') === 'true';
        var libraryRequest = 0;
        var horizon = null;
        var horizonWarnNoun = null;
        // True once the winter fields hold settings (loaded from a seasonal profile or copied from summer).
        var winterFilled = false;

        function field(id) { return document.getElementById(prefix + '-' + id); }
        function sourceRadios() { return container.querySelectorAll('input[name="' + prefix + '-source"]'); }
        function hasLibrary() { return Boolean(field('library-select')); }
        function seasonsOn() { return field('seasons_enabled').checked; }

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

        // shapePrefix: '' for the whole run (or summer), 'winter-' for winter.
        function syncShape(shapePrefix) {
            SEGMENTS.forEach(function (name) {
                var moments = usesMoments(field(shapePrefix + name + '-distribution').value);
                field(shapePrefix + name + '-mean-group').hidden = !moments;
                field(shapePrefix + name + '-std-group').hidden = !moments;
            });
            field(shapePrefix + 'night-text').textContent = nightText(
                numberOrText(field(shapePrefix + 'day_start_hour').value),
                numberOrText(field(shapePrefix + 'day_end_hour').value));
        }

        function syncSeasons() {
            var on = seasonsOn();
            field('season-fields').hidden = !on;
            field('winter').hidden = !on;
            field('shape-title').textContent = on ? 'Summer' : 'Day/Night';
            var summerEnd = numberOrText(field('summer_end_period').value);
            field('season-text').textContent = seasonText(summerEnd, horizon);
            var warning = field('season-warning');
            var wholeRun = on && horizonWarnNoun && isInteger(horizon) && isInteger(summerEnd) && summerEnd >= horizon;
            warning.hidden = !wholeRun;
            warning.textContent = wholeRun ? 'The whole ' + horizonWarnNoun + ' is summer.' : '';
        }

        function syncVisibility() {
            var source = getSource();
            field('generated').hidden = source === 'file';
            if (hasLibrary()) {
                field('library').hidden = source !== 'library';
                setReadOnly(source === 'library');
            }
            syncShape('');
            syncShape('winter-');
            syncSeasons();
        }

        function readShape(shapePrefix) {
            var shape = {
                day_start_hour: numberOrText(field(shapePrefix + 'day_start_hour').value),
                day_end_hour: numberOrText(field(shapePrefix + 'day_end_hour').value),
                morning_ramp_hours: numberOrZero(field(shapePrefix + 'morning_ramp_hours').value),
                evening_ramp_hours: numberOrZero(field(shapePrefix + 'evening_ramp_hours').value)
            };
            SEGMENTS.forEach(function (name) {
                var distribution = field(shapePrefix + name + '-distribution').value;
                var segment = { distribution: distribution };
                (usesMoments(distribution) ? ['mean', 'std', 'min', 'max'] : ['min', 'max']).forEach(function (key) {
                    segment[key] = numberOrText(field(shapePrefix + name + '-' + key).value);
                });
                shape[name] = segment;
            });
            return shape;
        }

        function readProfile() {
            var profile = {
                version: 1,
                seed: numberOrText(field('seed').value),
                autocorrelation: numberOrText(field('autocorrelation').value)
            };
            var shape = readShape('');
            Object.keys(shape).forEach(function (key) { profile[key] = shape[key]; });
            profile.seasons = null;
            if (seasonsOn()) {
                profile.seasons = {
                    summer_end_period: numberOrText(field('summer_end_period').value),
                    transition_days: numberOrZero(field('transition_days').value),
                    winter: readShape('winter-')
                };
            }
            return profile;
        }

        function writeShape(shapePrefix, shape) {
            shape = shape || {};
            ['day_start_hour', 'day_end_hour'].forEach(function (key) {
                var value = shape[key] !== undefined && shape[key] !== null ? shape[key] : defaults[key];
                field(shapePrefix + key).value = value;
            });
            RAMPS.forEach(function (pair) {
                var value = shape[pair[0]];
                field(shapePrefix + pair[0]).value = value !== undefined && value !== null ? value : 0;
            });
            SEGMENTS.forEach(function (name) {
                var segment = shape[name] || {};
                var fallback = defaults[name];
                field(shapePrefix + name + '-distribution').value = DISTRIBUTIONS.indexOf(segment.distribution) !== -1
                    ? segment.distribution : fallback.distribution;
                ['mean', 'std', 'min', 'max'].forEach(function (key) {
                    var value = segment[key] !== undefined && segment[key] !== null ? segment[key] : fallback[key];
                    field(shapePrefix + name + '-' + key).value = value;
                });
            });
        }

        // Fields of the whole-run (summer) shape as a shape object (all four moment fields, for copying).
        function rawShape(shapePrefix) {
            var shape = {};
            ['day_start_hour', 'day_end_hour', 'morning_ramp_hours', 'evening_ramp_hours'].forEach(function (key) {
                shape[key] = field(shapePrefix + key).value;
            });
            SEGMENTS.forEach(function (name) {
                var segment = { distribution: field(shapePrefix + name + '-distribution').value };
                ['mean', 'std', 'min', 'max'].forEach(function (key) {
                    segment[key] = field(shapePrefix + name + '-' + key).value;
                });
                shape[name] = segment;
            });
            return shape;
        }

        function writeProfile(profile) {
            field('seed').value = profile.seed;
            field('autocorrelation').value = profile.autocorrelation;
            writeShape('', profile);
            var seasons = isMapping(profile.seasons) ? profile.seasons : null;
            field('seasons_enabled').checked = seasons !== null;
            if (seasons !== null) {
                field('summer_end_period').value = seasons.summer_end_period !== undefined && seasons.summer_end_period !== null
                    ? seasons.summer_end_period : '';
                field('transition_days').value = seasons.transition_days !== undefined && seasons.transition_days !== null
                    ? seasons.transition_days : 0;
                writeShape('winter-', seasons.winter);
                winterFilled = true;
            } else {
                field('summer_end_period').value = '';
                field('transition_days').value = 0;
                writeShape('winter-', profile);
                winterFilled = false;
            }
        }

        // Switching the split on: winter starts as a copy of the summer settings, the last summer period
        // as half of the horizon (or 4380 h, half a year, when the horizon is unknown).
        function seasonsSwitched() {
            if (!seasonsOn()) { return; }
            if (!winterFilled) {
                writeShape('winter-', rawShape(''));
                winterFilled = true;
            }
            if (field('summer_end_period').value.trim() === '') {
                field('summer_end_period').value = isInteger(horizon) && horizon >= 2
                    ? Math.floor(horizon / 2) : 4380;
            }
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

        function setHorizon(periods, warnNoun) {
            horizon = isInteger(periods) && periods >= 1 ? periods : null;
            horizonWarnNoun = warnNoun || null;
            syncSeasons();
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

        // Error key (server or client) to the id suffix of its input: "day.mean" -> "day-mean",
        // "seasons.transition_days" -> "transition_days", "seasons.winter.night.std" -> "winter-night-std".
        function inputIdForKey(key) {
            if (key === 'demand_profile_source') { return hasLibrary() ? 'library' : 'source'; }
            var name = key.replace(/^(demand_profile|profile)\.?/, '');
            if (name === '' || name === 'version') { return 'source'; }
            if (name.indexOf('seasons.') === 0) { name = name.slice('seasons.'.length); }
            return name.replace(/\./g, '-');
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

        // Shows errors keyed "seed", "day.mean", "seasons.winter.day.mean" or "demand_profile.day.mean";
        // returns the number shown.
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
                if (event.target.id === prefix + '-seasons_enabled') {
                    seasonsSwitched();
                    setError('seasons', '');
                } else {
                    setError(event.target.id.slice(prefix.length + 1), '');
                }
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
            setHorizon: setHorizon,
            validate: validate,
            showErrors: showErrors,
            clearErrors: clearErrors,
            onChange: function (callback) { listeners.push(callback); },
            previewRequest: previewRequest
        };
    }

    // ------------------------------------------------------------------ preview

    var NIGHT_FILL = 'rgba(58, 65, 111, 0.10)';
    var SEASON_COLOR = '#17ad37';

    // Shades the night periods (preview.night_mask) behind the series and draws a dashed vertical line at the
    // end of the last summer period (preview.season_boundary).
    var nightShading = {
        id: 'demandNightShading',
        beforeDatasetsDraw: function (chart, args, options) {
            var x = chart.scales.x;
            var area = chart.chartArea;
            var ctx = chart.ctx;
            var mask = options && options.mask;
            ctx.save();
            if (mask && mask.length) {
                ctx.fillStyle = NIGHT_FILL;
                var index = 0;
                while (index < mask.length) {
                    if (!mask[index]) { index += 1; continue; }
                    var first = index;
                    while (index < mask.length && mask[index]) { index += 1; }
                    var left = Math.max(area.left, x.getPixelForValue(first + 0.5));
                    var right = Math.min(area.right, x.getPixelForValue(index + 0.5));
                    if (right > left) { ctx.fillRect(left, area.top, right - left, area.bottom - area.top); }
                }
            }
            var boundary = options && options.boundary;
            if (isNumber(boundary)) {
                var position = x.getPixelForValue(boundary + 0.5);
                if (position >= area.left && position <= area.right) {
                    ctx.strokeStyle = SEASON_COLOR;
                    ctx.lineWidth = 2;
                    ctx.setLineDash([6, 4]);
                    ctx.beginPath();
                    ctx.moveTo(position, area.top);
                    ctx.lineTo(position, area.bottom);
                    ctx.stroke();
                }
            }
            ctx.restore();
        }
    };

    function formatMWh(value) {
        return Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 });
    }

    // Season boundary inside the chart (a summer end before the last period), else null.
    function visibleBoundary(preview) {
        var boundary = preview.season_boundary;
        return isInteger(boundary) && boundary >= 1 && boundary < preview.periods ? boundary : null;
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
        var boundary = visibleBoundary(preview);
        canvasEl.setAttribute('aria-label', 'Demand of periods 1 to ' + preview.periods + ' ('
            + (preview.source === 'generated' ? 'generated' : 'demand file') + '): min ' + formatMWh(stats.min)
            + ', mean ' + formatMWh(stats.mean) + ', max ' + formatMWh(stats.max) + ' MWh. Night hours are shaded.'
            + (boundary !== null ? ' Summer ends after period ' + boundary + ' (dashed line).' : ''));
        var datasets = [{
            label: 'Demand (MWh)', data: points, borderColor: '#cb0c9f', backgroundColor: '#cb0c9f',
            pointRadius: points.length > 60 ? 0 : 2, borderWidth: 1.5, tension: 0
        }, {
            label: 'Night hours', data: [], borderColor: 'rgba(58, 65, 111, 0.25)',
            backgroundColor: 'rgba(58, 65, 111, 0.25)', pointStyle: 'rect'
        }];
        if (boundary !== null) {
            // Legend entry of the season boundary line drawn by the plugin.
            datasets.push({
                label: 'End of summer (period ' + boundary + ')', data: [], borderColor: SEASON_COLOR,
                backgroundColor: SEASON_COLOR, borderDash: [6, 4], pointStyle: 'line'
            });
        }
        return new global.Chart(canvasEl.getContext('2d'), {
            type: 'line',
            data: { datasets: datasets },
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
                    demandNightShading: { mask: preview.night_mask, boundary: boundary }
                }
            },
            plugins: [nightShading]
        });
    }

    function renderStats(tableBodyEl, preview) {
        tableBodyEl.textContent = '';
        if (!preview || !preview.stats) { return; }
        var stats = preview.stats;
        var rows = [['Overall', stats], ['Day', stats.day], ['Night', stats.night]];
        if ('summer' in stats || 'winter' in stats) {
            rows.push(['Summer', stats.summer], ['Winter', stats.winter]);
        }
        rows.forEach(function (pair) {
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
        seasonText: seasonText,
        renderPreview: renderPreview,
        renderStats: renderStats,
        clearPreview: clearPreview,
        createLoader: createLoader,
        DEFAULT_PROFILE: copy(DEFAULT_PROFILE)
    };
})(window);
