/*
 * Shared chart tools of the analysis tab (PLAN 2.11, spec watch_analysis_ui).
 *
 * Exposes window.ChartTools:
 *   - create(canvas, config): builds a Chart.js 3.7.1 chart with wheel zoom,
 *     pinch zoom and drag-pan on the x axis (chartjs-plugin-zoom 1.2.1 with
 *     hammerjs 2.0.8, loaded before this file),
 *   - copyConfig(config): deep copy of a chart config (plain objects and
 *     arrays are copied, functions and plugin objects are shared),
 *   - resetZoom(chart),
 *   - openFullscreen(options): shows a larger copy of a chart in a Bootstrap
 *     5 modal-fullscreen with its own reset and PNG buttons. Closing the
 *     modal destroys the copy.
 *   - bindButtons(root, handlers): wires the [data-zoom-reset] and
 *     [data-expand] buttons inside root.
 *
 * Without the zoom plugin the charts are built without zoom and the reset
 * buttons do nothing.
 */
(function (global) {
  'use strict';

  function zoomAvailable() {
    return typeof global.Chart !== 'undefined' && !!global.Chart.registry &&
      (function () {
        try { return !!global.Chart.registry.getPlugin('zoom'); } catch (error) { return false; }
      })();
  }

  function isPlainObject(value) {
    if (value === null || typeof value !== 'object') { return false; }
    // Plain objects have Object.prototype (whose prototype is null) or null as
    // prototype. The check does not depend on the realm of Object.prototype.
    var proto = Object.getPrototypeOf(value);
    return proto === null || Object.getPrototypeOf(proto) === null;
  }

  function copyConfig(value) {
    if (Array.isArray(value)) { return value.map(copyConfig); }
    if (!isPlainObject(value)) { return value; }
    var out = {};
    Object.keys(value).forEach(function (key) { out[key] = copyConfig(value[key]); });
    return out;
  }

  function zoomOptions() {
    return {
      pan: { enabled: true, mode: 'x' },
      zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
    };
  }

  function withZoom(config) {
    if (!zoomAvailable()) { return config; }
    config.options = config.options || {};
    config.options.plugins = config.options.plugins || {};
    config.options.plugins.zoom = zoomOptions();
    return config;
  }

  function create(canvas, config) {
    return new global.Chart(canvas.getContext('2d'), withZoom(config));
  }

  function resetZoom(chart) {
    if (chart && typeof chart.resetZoom === 'function') { chart.resetZoom(); }
  }

  function savePng(chart, name) {
    if (!chart) { return; }
    var link = document.createElement('a');
    link.href = chart.toBase64Image('image/png');
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  var modalChart = null;
  var modalBound = false;

  function modalElements() {
    var modal = document.getElementById('an-chart-modal');
    if (!modal) { return null; }
    return {
      modal: modal,
      title: document.getElementById('an-chart-modal-title'),
      canvas: document.getElementById('an-chart-modal-canvas'),
      reset: document.getElementById('an-chart-modal-reset'),
      png: document.getElementById('an-chart-modal-png')
    };
  }

  function destroyModalChart() {
    if (modalChart) { modalChart.destroy(); modalChart = null; }
  }

  // options: { title, config (a copy owned by the modal), fileName }
  function openFullscreen(options) {
    var els = modalElements();
    if (!els || typeof global.bootstrap === 'undefined') { return; }
    if (!modalBound) {
      modalBound = true;
      els.modal.addEventListener('hidden.bs.modal', destroyModalChart);
      els.reset.addEventListener('click', function () { resetZoom(modalChart); });
    }
    destroyModalChart();
    els.title.textContent = options.title || '';
    els.png.onclick = function () { savePng(modalChart, options.fileName || 'chart.png'); };
    var instance = global.bootstrap.Modal.getOrCreateInstance(els.modal);
    var build = function () {
      els.modal.removeEventListener('shown.bs.modal', build);
      destroyModalChart();
      var config = copyConfig(options.config);
      config.options = config.options || {};
      config.options.maintainAspectRatio = false;
      modalChart = create(els.canvas, config);
    };
    if (els.modal.classList.contains('show')) {
      build();
    } else {
      els.modal.addEventListener('shown.bs.modal', build);
      instance.show();
    }
  }

  // handlers: { chart(key), config(key), title(key), fileName(key) }
  function bindButtons(root, handlers) {
    Array.prototype.forEach.call(root.querySelectorAll('[data-zoom-reset]'), function (button) {
      button.addEventListener('click', function () {
        resetZoom(handlers.chart(button.getAttribute('data-zoom-reset')));
      });
    });
    Array.prototype.forEach.call(root.querySelectorAll('[data-expand]'), function (button) {
      button.addEventListener('click', function () {
        var key = button.getAttribute('data-expand');
        var config = handlers.config(key);
        if (!config) { return; }
        openFullscreen({ title: handlers.title(key), config: config, fileName: handlers.fileName(key) });
      });
    });
  }

  global.ChartTools = {
    create: create,
    copyConfig: copyConfig,
    resetZoom: resetZoom,
    openFullscreen: openFullscreen,
    bindButtons: bindButtons,
    zoomAvailable: zoomAvailable
  };
})(window);
