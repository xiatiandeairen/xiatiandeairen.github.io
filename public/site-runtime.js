/* site-runtime.js — Shared client runtime for the site.
   Loaded once via <script src="/site-runtime.js"> in BaseLayout, exposes
   window.SiteRuntime with helpers used by inline IIFEs across pages.
   Single source of truth for: HTML escaping, search query parsing/scoring,
   date filter, snippet extraction. Mirrors src/utils/search.ts behavior. */
(function() {
  'use strict';

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function(c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function highlight(text, term) {
    if (!term) return escapeHtml(text);
    return escapeHtml(text).replace(
      new RegExp(escapeRe(escapeHtml(term)), 'gi'),
      '<mark>$&</mark>'
    );
  }

  var PREFIX_RE = /\b(tag|topic|type|series|date):([^\s]+)/gi;

  function parseQuery(raw) {
    var out = { free: '' };
    var free = raw || '';
    var m;
    PREFIX_RE.lastIndex = 0;
    while ((m = PREFIX_RE.exec(raw || '')) !== null) {
      out[m[1].toLowerCase()] = m[2].toLowerCase();
      free = free.replace(m[0], '');
    }
    out.free = free.trim().toLowerCase();
    return out;
  }

  function parseDateSpec(spec) {
    if (!spec) return null;
    var now = new Date();
    var m;
    if ((m = spec.match(/^(\d+)d$/))) {
      var s = new Date(now); s.setDate(s.getDate() - parseInt(m[1], 10));
      return { since: s.toISOString(), until: now.toISOString() };
    }
    if ((m = spec.match(/^(\d+)y$/))) {
      var s2 = new Date(now); s2.setFullYear(s2.getFullYear() - parseInt(m[1], 10));
      return { since: s2.toISOString(), until: now.toISOString() };
    }
    if ((m = spec.match(/^(\d{4})$/))) {
      return { since: m[1] + '-01-01T00:00:00Z', until: m[1] + '-12-31T23:59:59Z' };
    }
    if ((m = spec.match(/^(\d{4})-(\d{2})$/))) {
      var y = parseInt(m[1], 10), mo = parseInt(m[2], 10);
      var last = new Date(y, mo, 0).getDate();
      return {
        since: m[1] + '-' + m[2] + '-01T00:00:00Z',
        until: m[1] + '-' + m[2] + '-' + String(last).padStart(2, '0') + 'T23:59:59Z'
      };
    }
    return null;
  }

  function snippetOf(content, term) {
    if (!content) return '';
    if (!term) return content.slice(0, 160);
    var idx = content.toLowerCase().indexOf(term);
    if (idx < 0) return content.slice(0, 160);
    var start = Math.max(0, idx - 60);
    var end = Math.min(content.length, idx + term.length + 100);
    return (start > 0 ? '…' : '') + content.slice(start, end) + (end < content.length ? '…' : '');
  }

  /* searchNotes: mirror of src/utils/search.ts::searchNotes.
     Index schema v1.1 fields per note: slug, title, content, tags, tagAliases,
     topics, topicAliases, questionType, series, date. */
  function searchNotes(rawQuery, index, limit) {
    var q = parseQuery(rawQuery);
    if (!q.free && !q.tag && !q.topic && !q.type && !q.series && !q.date) return [];
    var dateRange = q.date ? parseDateSpec(q.date) : null;
    var hits = [];
    for (var i = 0; i < index.notes.length; i++) {
      var n = index.notes[i];
      if (q.tag) {
        var tHit = n.tags.some(function(t) { return t.toLowerCase().includes(q.tag); });
        var taHit = (n.tagAliases || []).some(function(a) { return a.toLowerCase().includes(q.tag); });
        if (!tHit && !taHit) continue;
      }
      if (q.topic) {
        var tpHit = n.topics.some(function(t) { return t.toLowerCase().includes(q.topic); });
        var tpaHit = (n.topicAliases || []).some(function(a) { return a.toLowerCase().includes(q.topic); });
        if (!tpHit && !tpaHit) continue;
      }
      if (q.type && !(n.questionType || '').toLowerCase().includes(q.type)) continue;
      if (q.series && !((n.series || '').toLowerCase().includes(q.series))) continue;
      if (dateRange && (n.date < dateRange.since || n.date > dateRange.until)) continue;

      var score = 0, matched = 'content', free = q.free;
      if (!free) {
        score = 50;
        matched = q.tag ? 'tag' : q.topic ? 'topic' : q.series ? 'series' : q.type ? 'type' : 'content';
      } else {
        var title = n.title.toLowerCase();
        var content = (n.content || '').toLowerCase();
        var tagsLow = n.tags.map(function(t) { return t.toLowerCase(); });
        var topicsLow = n.topics.map(function(t) { return t.toLowerCase(); });
        var aliasesLow = ((n.tagAliases || []).concat(n.topicAliases || [])).map(function(a) { return a.toLowerCase(); });
        var sLow = (n.series || '').toLowerCase();

        if (title === free) { score = 100; matched = 'title'; }
        else if (title.includes(free)) { score = 60; matched = 'title'; }
        else if (tagsLow.indexOf(free) >= 0) { score = 55; matched = 'tag'; }
        else if (topicsLow.indexOf(free) >= 0) { score = 55; matched = 'topic'; }
        else if (sLow === free) { score = 50; matched = 'series'; }
        else if (tagsLow.some(function(t) { return t.includes(free); })) { score = 35; matched = 'tag'; }
        else if (topicsLow.some(function(t) { return t.includes(free); })) { score = 35; matched = 'topic'; }
        else if (sLow.includes(free)) { score = 30; matched = 'series'; }
        else if (aliasesLow.some(function(a) { return a.includes(free); })) { score = 25; matched = 'alias'; }
        else if (content.includes(free)) { score = 10; matched = 'content'; }
        else if (q.tag || q.topic || q.type || q.series) { score = 20; }
        else { continue; }
      }
      hits.push({
        note: n,
        score: score,
        matchedField: matched,
        snippet: snippetOf(n.content || '', free || q.tag || q.topic || q.series || q.type || '')
      });
    }
    hits.sort(function(a, b) {
      if (b.score !== a.score) return b.score - a.score;
      return a.note.date < b.note.date ? 1 : a.note.date > b.note.date ? -1 : 0;
    });
    return limit ? hits.slice(0, limit) : hits;
  }

  /* Index version guard: warn (don't crash) if cached index predates current schema. */
  function indexVersionWarning(index) {
    var v = index && index.version;
    if (!v || v < '1.1') {
      try { console.warn('[search] cached index version', v, '< 1.1; some features may degrade. Hard reload to refresh.'); } catch (e) {}
      return false;
    }
    return true;
  }

  window.SiteRuntime = {
    escapeHtml: escapeHtml,
    escapeRe: escapeRe,
    highlight: highlight,
    parseQuery: parseQuery,
    parseDateSpec: parseDateSpec,
    snippetOf: snippetOf,
    searchNotes: searchNotes,
    indexVersionWarning: indexVersionWarning,
  };
})();
