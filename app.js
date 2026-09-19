(function () {
  // The payload is its own file, fetched once and cached across every
  // page. Inlining it put 49KB of identical JSON in eight documents.
  var D = window.QAL;
  var SLIP_KEY = "qal.slip";
  var picked = [];
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (t) { var d = document.createElement("div"); d.textContent = t == null ? "" : t; return d.innerHTML; };
  var pct = function (x) { return x == null ? "\u2014" : Math.round(x * 100) + "%"; };
  var sign = function (x) { return (x > 0 ? "+" : "") + x.toFixed(1); };

  // ---- nav counts
  // No click handler any more: each view is its own document and the nav is
  // eight ordinary links, so the browser handles what this used to fake.
  var counts = { edges: D.edges.length, board: D.card.length, matchups: D.matchups.length,
                 grid: D.ranks.length, tds: D.tds.length, feed: D.feed.length,
                 search: D.edges.length, slip: 0 };
  document.querySelectorAll("[data-count]").forEach(function (s) {
    var n = counts[s.dataset.count];
    if (n) s.textContent = n;
  });

  // ---- edges
  function edgeRow(x, i) {
    var l5 = "";
    if (x.last5) {
      l5 = '<div><div class="lbl">Last ' + x.last5.n + '</div><div class="last5">' +
        x.last5.hits.map(function (h) { return '<i class="' + (h ? "hit" : "") + '"></i>'; }).join("") +
        "</div></div>";
    }
    var sub = x.last5
      ? '<div class="edge-sub">' + x.last5.hits.filter(Boolean).length + " of " + x.last5.n +
        " cleared it &middot; " + x.last5.avg + " avg vs " + x.line + "</div>"
      : "";
    var alt = x.alt
      ? '<div class="alt"><div class="alt-head"><span class="lbl">Safer alternate</span>' +
        '<span class="p">' + (x.alt.price > 0 ? "+" : "") + x.alt.price + "</span></div>" +
        '<div class="l">Over ' + x.alt.line + " " + esc(x.market.toLowerCase()) + "</div>" +
        '<div class="m">' + esc(x.alt.book) + " &middot; " + (x.line - x.alt.line).toFixed(1) +
        " lower</div></div>"
      : "";
    var oneon = '<div><div class="lbl">The matchup</div><div class="oneon">' +
      '<span class="who">vs ' + esc(x.opp) + " defence</span><span class='rule'></span>" +
      '<span class="rank">' + (x.def_rank ? "ranked " + x.def_rank + " of 32" : "rank unavailable") +
      "</span></div></div>";
    return '<div class="edge"><div class="edge-top"><div>' +
      '<span class="edge-who">' + esc(x.player) + '</span>' +
      '<span class="edge-meta">' + esc(x.pos) + " &middot; " + esc(x.team) + " vs " + esc(x.opp) + "</span>" +
      '<div class="edge-line">Over ' + x.line + " " + esc(x.market.toLowerCase()) + "</div></div>" +
      '<div class="edge-gap"><div class="v">' + sign(x.gap) + '</div><div class="lbl">Gap</div></div></div>' +
      '<div class="edge-stats">' +
      (x.mp != null ? '<div><div class="lbl">Model</div><div class="v">' + pct(x.mp) + "</div></div>" : "") +
      (x.implied != null ? '<div><div class="lbl">Implied</div><div class="v">' + pct(x.implied) + "</div></div>" : "") +
      (x.price != null ? '<div><div class="lbl">Price</div><div class="v">' + (x.price > 0 ? "+" : "") + x.price + "</div></div>" : "") +
      (x.ev != null ? '<div><div class="lbl">EV</div><div class="v">' + (x.ev >= 0 ? "+" : "") + (x.ev * 100).toFixed(1) + "%</div></div>" : "") +
      '<div><div class="lbl">Ours</div><div class="v">' + x.proj + "</div></div>" +
      l5 + "</div>" + sub +
      '<div class="edge-detail">' + oneon + alt + "</div>" +
      '<button class="addbtn" data-add="' + i + '" aria-pressed="false">Add to slip</button></div>';
  }

  function renderEdges() {
    var min = parseFloat($("minGap").value) || 0;
    var rows = D.edges.filter(function (x) { return Math.abs(x.gap) >= min; });
    $("minGapLabel").textContent = min;
    $("edgeCount").textContent = rows.length + " of " + D.edges.length;
    $("edgeList").innerHTML = rows.length
      ? rows.map(function (x) { return edgeRow(x, D.edges.indexOf(x)); }).join("")
      : '<div class="empty">Nothing clears a ' + min + ' point gap. Lower the floor to see thinner edges.</div>';
    wireAdds();
  }
  if (!$("edgeList")) { /* not this page */ }
  else if (!D.edges.length) {
    $("edgeList").innerHTML = '<div class="empty">No prop lines posted yet. Books put these up close to kickoff \u2014 check back nearer the weekend.</div>';
    $("edgeCount").textContent = "0";
  } else {
    $("minGap").addEventListener("input", renderEdges);
    renderEdges();
  }

  // ---- board
  if ($("boardRows")) $("boardRows").innerHTML = D.card.length ? D.card.map(function (c) {
    return "<tr><td><span class='team'>" + esc(c.away) + " at " + esc(c.home) +
      "</span><span class='when'>" + esc(c.kick) + "</span></td>" +
      "<td class='num'>" + (c.price > 0 ? "+" : "") + c.price + "</td>" +
      "<td class='num'>" + pct(c.implied) + "</td>" +
      "<td class='model'>" + esc(c.pick) + "</td>" +
      "<td class='model'>" + pct(c.conf) + "</td>" +
      "<td class='num'>" + c.total + "</td></tr>";
  }).join("") : "<tr><td colspan='7' class='soft'>No lines posted yet.</td></tr>";

  // ---- matchups
  if ($("matchupList")) $("matchupList").innerHTML = D.matchups.length ? D.matchups.map(function (m) {
    var r = m.def_rank;
    var w = r ? (33 - r) / 32 * 100 : 0;
    return '<div class="edge"><div><span class="edge-who">' + esc(m.player) + "</span> " +
      '<span class="edge-meta">' + esc(m.pos) + " &middot; " + esc(m.team) + " vs " + esc(m.opp) +
      (r ? " &middot; defence ranked " + r : " &middot; no rank") + '</span>' +
      '<div class="range"><i style="left:0;width:' + w + '%"></i></div></div>' +
      '<div class="edge-nums"><div><span class="k">Line</span><span class="v">' + m.line +
      '</span></div><div><span class="k">Ours</span><span class="v">' + m.proj +
      '</span></div><div><span class="k">Gap</span><span class="v gap ' +
      (m.gap >= 0 ? "pos" : "neg") + '">' + sign(m.gap) + "</span></div></div></div>";
  }).join("") : '<div class="empty">Needs prop lines and team ranks. Both arrive closer to kickoff.</div>';

  // ---- grid
  var RAMP = ["#141414","#3D3D3D","#666666","#8E8E8E","#B4B4B4","#D6D6D6","#EBEBEB"];
  function shade(rank) {
    if (rank == null) return { bg: "#F6F6F6", fg: "#9A9A9A" };
    var i = Math.min(RAMP.length - 1, Math.floor((rank - 1) / 32 * RAMP.length));
    return { bg: RAMP[i], fg: i <= 2 ? "#fff" : "#141414" };
  }
  function cell(rank) {
    var c = shade(rank);
    return '<td class="rank"><span class="pill" style="background:' + c.bg + ';color:' + c.fg +
      '">' + (rank == null ? "\u2014" : rank) + "</span></td>";
  }
  if ($("gridCells")) $("gridCells").innerHTML = D.ranks.length
    ? '<div class="scroller"><table class="gridtable"><thead><tr><th>Team</th>' +
      '<th class="num">Pass off</th><th class="num">Rush off</th>' +
      '<th class="num">Def (EPA)</th><th class="num">Pts off</th></tr></thead><tbody>' +
      D.ranks.map(function (r) {
        return "<tr><td class='team'>" + esc(r.name || r.team) + "</td>" +
          cell(r.pass_off) + cell(r.rush_off) + cell(r.total_def) + cell(r.points_off) + "</tr>";
      }).join("") + "</tbody></table></div>"
    : '<div class="empty">Team ranks unavailable.</div>';

  // ---- touchdowns
  function renderTds() {
    var pos = $("tdPos").value;
    var rows = D.tds.filter(function (t) { return !pos || t.pos === pos; });
    $("tdCount").textContent = rows.length + " of " + D.tds.length
      + (D.tds_removed ? "  ·  " + D.tds_removed + " removed as Out or IR" : "");
    if (!D.tds.length) {
      $("tdList").innerHTML = '<div class="empty">No touchdown projections in this build. '
        + 'They need the usage pipeline, which runs only in a full build.</div>';
      return;
    }
    $("tdList").innerHTML = rows.length ? rows.map(function (t) {
      var flag = t.inj ? " <span class='inj'>" + esc(t.inj) + "</span>" : "";
      return '<div class="feeditem"><span class="when2">' + esc(String(t.prob)) + "%</span>" +
        "<div><div class='hl'><span class='team'>" + esc(t.team) + "</span>" + esc(t.player) +
        " <span class='muted'>" + esc(t.pos) + " vs " + esc(t.opp) + "</span>" + flag + "</div></div>" +
        "<span class='tag'>" + (t.fair > 0 ? "+" : "") + esc(String(t.fair)) + "</span></div>";
    }).join("") : '<div class="empty">No players at that position.</div>';
  }
  if ($("tdList")) { $("tdPos").addEventListener("change", renderTds); renderTds(); }

  // ---- feed
  // Only clubs that actually have items. Listing all 32 with 29 of them empty
  // makes the control look broken.
  (function () {
    var seen = {};
    D.feed.forEach(function (f) { if (f.team) seen[f.team] = (seen[f.team] || 0) + 1; });
    var sel = $("feedTeam");
    if (!sel) return;
    Object.keys(seen).sort().forEach(function (t) {
      var o = document.createElement("option");
      o.value = t; o.textContent = t + " (" + seen[t] + ")";
      sel.appendChild(o);
    });
  })();

  function renderFeed() {
    var team = $("feedTeam").value;
    var cardOnly = $("feedCard").checked;
    var rows = D.feed.filter(function (f) {
      if (team && f.team !== team) return false;
      if (cardOnly && !f.on_card) return false;
      return true;
    });
    $("feedCount").textContent = rows.length + " of " + D.feed.length;
    if (!rows.length) {
      $("feedList").innerHTML = '<div class="empty">' + (D.feed.length
        ? "Nothing matches that filter. ESPN's news feed carries no club field, so items mentioning two clubs or none are left unassigned."
        : "No injury or news items right now.") + '</div>';
      return;
    }
    $("feedList").innerHTML = rows.map(function (f) {
      return '<div class="feeditem"><span class="when2">' + esc(f.time) + "</span>" +
        "<div><div class='hl'>" +
        (f.team ? "<span class='team'>" + esc(f.team) + "</span>" : "") +
        esc(f.headline) + "</div>" +
        "<div class='dt'>" + esc(f.detail) + "</div>" +
        (f.touches ? "<div class='touchline'>Touches a player on the board</div>" : "") +
        "</div><span class='tag'>" + (f.kind === "injury" ? "Injury" : "News") + "</span></div>";
    }).join("");
  }
  if ($("feedList")) {
    $("feedTeam").addEventListener("change", renderFeed);
    $("feedCard").addEventListener("change", renderFeed);
    renderFeed();
  }

  // ---- search
  function renderSearch() {
    var q = ($("q").value || "").toLowerCase().trim();
    if (!q) { $("searchList").innerHTML = '<div class="empty">Type a name to see his line and our number.</div>'; return; }
    var hits = D.edges.filter(function (x) { return x.player.toLowerCase().indexOf(q) !== -1; });
    $("searchList").innerHTML = hits.length
      ? hits.map(function (x) { return edgeRow(x, D.edges.indexOf(x)); }).join("")
      : '<div class="empty">No one on the board matches that.</div>';
    wireAdds();
  }
  if ($("searchList")) { $("q").addEventListener("input", renderSearch); renderSearch(); }

  // ---- slip
  function legId(x) { return x.player + "|" + x.market + "|" + x.line; }
  function loadSlip() {
    try {
      var ids = JSON.parse(sessionStorage.getItem(SLIP_KEY) || "[]");
      var out = [];
      D.edges.forEach(function (x, i) { if (ids.indexOf(legId(x)) !== -1) out.push(i); });
      return out;
    } catch (e) { return []; }   // private mode, blocked storage, stale JSON
  }
  function saveSlip() {
    try {
      sessionStorage.setItem(SLIP_KEY, JSON.stringify(
        picked.map(function (i) { return legId(D.edges[i]); })));
    } catch (e) { /* the slip still works, it just will not follow a link */ }
  }
  function dec(american) { return american < 0 ? 1 + 100 / Math.abs(american) : 1 + american / 100; }
  function amer(d) { return d >= 2 ? "+" + Math.round((d - 1) * 100) : String(Math.round(-100 / (d - 1))); }
  function renderSlip() {
    $("slip").hidden = picked.length === 0;
    var prob = 1, dd = 1, priced = true;
    $("slipLegs").innerHTML = "";
    picked.forEach(function (i) {
      var x = D.edges[i];
      // OUR probability, never the book's implied one. Multiplying implied by
      // the book's own odds returns 1 minus the vig by construction, so every
      // slip would read an edge of about zero however good the legs were.
      if (x.price == null || x.mp == null) priced = false;
      else { dd *= dec(x.price); prob *= x.mp; }
      var li = document.createElement("li");
      li.innerHTML = "<strong>" + esc(x.player) + "</strong> <span class='soft'>" +
        esc(x.market) + " " + x.line + "</span>";
      var b = document.createElement("button");
      b.type = "button"; b.className = "x"; b.textContent = "\u00d7";
      b.setAttribute("aria-label", "Remove " + x.player);
      b.onclick = function () { toggle(i); };
      li.appendChild(b); $("slipLegs").appendChild(li);
    });
    $("slipProb").textContent = priced ? pct(prob) : "\u2014";
    $("slipOdds").textContent = priced ? amer(dd) : "\u2014";
    var edge = priced ? prob * dd - 1 : null;
    $("slipEdge").textContent = edge == null ? "\u2014" : (edge >= 0 ? "+" : "") + (edge * 100).toFixed(1) + "%";
    $("slipEdge").className = "v " + (edge != null && edge < 0 ? "neg" : "pos");
    var st = parseFloat($("stake").value) || 0;
    $("stakeOut").textContent = priced ? "returns " + (st * dd).toFixed(2) + " if it lands" : "needs prices";
    if ($("slipView")) {
      $("slipView").innerHTML = picked.length
        ? '<div class="scroller"><table><thead><tr><th>Leg</th><th class="num">Line</th>' +
          '<th class="num">Ours</th><th class="num">Model %</th><th class="num">Price</th></tr></thead><tbody>' +
          picked.map(function (i) {
            var x = D.edges[i];
            return "<tr><td><span class='team'>" + esc(x.team) + "</span>" + esc(x.player) +
              " <span class='soft'>" + esc(x.market.toLowerCase()) + "</span></td>" +
              "<td class='num'>" + x.line + "</td><td class='num'>" + x.proj + "</td>" +
              "<td class='num'>" + pct(x.mp) + "</td><td class='num'>" +
              (x.price == null ? "\u2014" : (x.price > 0 ? "+" : "") + x.price) + "</td></tr>";
          }).join("") + "</tbody></table></div>"
        : '<div class="empty">Nothing tracked yet. Add a leg from <a href="./">Best edges</a> ' +
          'or <a href="search.html">Player search</a> and it will follow you across the site.</div>';
    }
  }
  function toggle(i) {
    var at = picked.indexOf(i);
    if (at === -1) picked.push(i); else picked.splice(at, 1);
    saveSlip();
    document.querySelectorAll('[data-add="' + i + '"]').forEach(function (b) {
      b.setAttribute("aria-pressed", at === -1 ? "true" : "false");
      b.textContent = at === -1 ? "Added" : "Add";
    });
    renderSlip();
  }
  function wireAdds() {
    document.querySelectorAll("[data-add]").forEach(function (b) {
      var i = parseInt(b.dataset.add, 10);
      b.setAttribute("aria-pressed", picked.indexOf(i) === -1 ? "false" : "true");
      b.textContent = picked.indexOf(i) === -1 ? "Add" : "Added";
      b.onclick = function () { toggle(i); };
    });
  }
  picked = loadSlip();
  wireAdds();
  $("stake").addEventListener("input", renderSlip);
  $("slipClear").addEventListener("click", function () { picked.slice().forEach(toggle); });
  renderSlip();
})();
