/*
 * zip-store.js - minimal dependency-free ZIP writer (PLAN 2.5).
 *
 * Writes a ZIP archive with "stored" (uncompressed, method 0) entries as
 * specified in the PKWARE APPNOTE (section 4.3): one local file header plus
 * data per entry, the central directory and the end of central directory
 * record. File names are UTF-8 (general purpose flag bit 11). CRC-32 uses the
 * reflected polynomial 0xEDB88320. ZIP64 is not supported, so an archive is
 * limited to 65535 entries and 4 GiB, far above a set of chart PNGs.
 *
 * API (global ZipStore, also module.exports under CommonJS):
 *   ZipStore.crc32(bytes) -> unsigned 32-bit integer
 *   ZipStore.build(entries, date?) -> Uint8Array, entries = [{name, data: Uint8Array}]
 *   ZipStore.toBlob(entries, date?) -> Blob of type application/zip
 *   ZipStore.dataUrlToBytes(dataUrl) -> Uint8Array (base64 data URL)
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else {
    root.ZipStore = api;
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var CRC_TABLE = (function () {
    var table = new Uint32Array(256);
    for (var n = 0; n < 256; n += 1) {
      var c = n;
      for (var k = 0; k < 8; k += 1) {
        c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      }
      table[n] = c >>> 0;
    }
    return table;
  }());

  function crc32(bytes) {
    var crc = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i += 1) {
      crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function utf8(text) {
    if (typeof TextEncoder !== 'undefined') {
      return new TextEncoder().encode(text);
    }
    var binary = unescape(encodeURIComponent(text));
    var out = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i += 1) { out[i] = binary.charCodeAt(i); }
    return out;
  }

  // MS-DOS time and date fields of a local Date (APPNOTE 4.4.6, 2 s resolution).
  function dosDateTime(date) {
    var year = Math.min(Math.max(date.getFullYear(), 1980), 2107);
    var time = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
    var day = ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
    return { time: time & 0xFFFF, date: day & 0xFFFF };
  }

  function build(entries, date) {
    if (!entries || !entries.length) { throw new Error('ZipStore.build needs at least one entry'); }
    if (entries.length > 0xFFFF) { throw new Error('ZipStore supports at most 65535 entries'); }
    var stamp = dosDateTime(date || new Date());
    var FLAG_UTF8 = 0x0800;
    var VERSION = 20;
    var prepared = entries.map(function (entry) {
      if (!entry || typeof entry.name !== 'string' || !entry.name) { throw new Error('ZipStore entry needs a name'); }
      var data = entry.data instanceof Uint8Array ? entry.data : new Uint8Array(entry.data);
      return { name: utf8(entry.name), data: data, crc: crc32(data) };
    });
    var localSize = 0;
    var centralSize = 0;
    prepared.forEach(function (p) {
      localSize += 30 + p.name.length + p.data.length;
      centralSize += 46 + p.name.length;
    });
    var total = localSize + centralSize + 22;
    if (total > 0xFFFFFFFF) { throw new Error('ZipStore archive exceeds 4 GiB'); }
    var out = new Uint8Array(total);
    var view = new DataView(out.buffer);
    var pos = 0;

    function u16(value) { view.setUint16(pos, value, true); pos += 2; }
    function u32(value) { view.setUint32(pos, value >>> 0, true); pos += 4; }
    function bytes(value) { out.set(value, pos); pos += value.length; }

    prepared.forEach(function (p) {
      p.offset = pos;
      u32(0x04034B50);            // local file header signature
      u16(VERSION);               // version needed to extract
      u16(FLAG_UTF8);             // general purpose bit flag
      u16(0);                     // compression method: stored
      u16(stamp.time);
      u16(stamp.date);
      u32(p.crc);
      u32(p.data.length);         // compressed size
      u32(p.data.length);         // uncompressed size
      u16(p.name.length);
      u16(0);                     // extra field length
      bytes(p.name);
      bytes(p.data);
    });
    var centralOffset = pos;
    prepared.forEach(function (p) {
      u32(0x02014B50);            // central directory header signature
      u16(VERSION);               // version made by
      u16(VERSION);               // version needed to extract
      u16(FLAG_UTF8);
      u16(0);                     // stored
      u16(stamp.time);
      u16(stamp.date);
      u32(p.crc);
      u32(p.data.length);
      u32(p.data.length);
      u16(p.name.length);
      u16(0);                     // extra field length
      u16(0);                     // file comment length
      u16(0);                     // disk number start
      u16(0);                     // internal file attributes
      u32(0);                     // external file attributes
      u32(p.offset);              // relative offset of the local header
      bytes(p.name);
    });
    var centralLength = pos - centralOffset;
    u32(0x06054B50);              // end of central directory signature
    u16(0);                       // number of this disk
    u16(0);                       // disk with the central directory
    u16(prepared.length);         // entries on this disk
    u16(prepared.length);         // total entries
    u32(centralLength);           // size of the central directory
    u32(centralOffset);           // offset of the central directory
    u16(0);                       // comment length
    return out;
  }

  function toBlob(entries, date) {
    return new Blob([build(entries, date)], { type: 'application/zip' });
  }

  function dataUrlToBytes(dataUrl) {
    var comma = dataUrl.indexOf(',');
    if (comma < 0 || dataUrl.slice(0, comma).indexOf(';base64') < 0) {
      throw new Error('ZipStore.dataUrlToBytes expects a base64 data URL');
    }
    var binary = atob(dataUrl.slice(comma + 1));
    var out = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i += 1) { out[i] = binary.charCodeAt(i); }
    return out;
  }

  return { crc32: crc32, build: build, toBlob: toBlob, dataUrlToBytes: dataUrlToBytes };
}));
