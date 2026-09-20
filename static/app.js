/**
 * Cyclone AI Frontend JavaScript App.
 * Connects to Python API endpoints and renders Chart.js charts and real-time inference data.
 */
document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentSampleName = null;
  let chartProb = null;
  let chartRI = null;
  let chartForecast = null;
  let chartComposition = null;
  let chartTraining = null;

  // IMD Category Palette -- single-hue sequential blue ramp for the 8 ordered
  // intensity categories (matches NHC/IMD wind-probability map convention),
  // with a reserved orange for the predicted/forecast highlight.
  const SEQ_RAMP = ["#f0f9ff", "#e0f2fe", "#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7", "#075985"];
  const CAT_ORANGE = "#ea580c";
  const CAT_BLUE = "#0369a1";
  const ICON_ALERT_TRIANGLE = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';

  // Global Chart.js theme -- match the app's own type + tooltip styling
  // instead of Chart.js's Helvetica-ish defaults, across every chart.
  Chart.defaults.font.family = "'Inter', 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif";
  Chart.defaults.font.size = 12.5;
  Chart.defaults.color = "#64748b";
  Chart.defaults.plugins.tooltip.backgroundColor = "#0f172a";
  Chart.defaults.plugins.tooltip.titleFont = { family: Chart.defaults.font.family, weight: "700", size: 12.5 };
  Chart.defaults.plugins.tooltip.bodyFont = { family: Chart.defaults.font.family, size: 12.5 };
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.displayColors = false;

  // Draws a value label above one highlighted bar (used for the predicted
  // category in the probabilities chart) -- a single custom plugin instead
  // of pulling in chartjs-plugin-datalabels for one label.
  Chart.register({
    id: "highlightLabel",
    afterDatasetsDraw(chart) {
      const opts = chart.config.options.plugins && chart.config.options.plugins.highlightLabel;
      if (!opts || !opts.enabled) return;
      const meta = chart.getDatasetMeta(0);
      const bar = meta.data[opts.index];
      if (!bar) return;
      const value = chart.data.datasets[0].data[opts.index];
      const { ctx } = chart;
      ctx.save();
      ctx.fillStyle = opts.color || "#0f172a";
      ctx.font = "700 13px 'Inter', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(opts.formatter ? opts.formatter(value) : value, bar.x, bar.y - 10);
      ctx.restore();
    }
  });

  // Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const topbarTitle = document.getElementById("topbar-title");
  const topbarSub = document.getElementById("topbar-sub");
  const TAB_META = {
    landing: { title: "Overview", sub: "Real-time tropical cyclone intelligence platform" },
    assess: { title: "Assess a Storm", sub: "Analyze satellite imagery and estimate cyclone intensity in real time" },
    history: { title: "Historical Precedent", sub: "Find historical cyclones with matching cloud structure" },
    ri: { title: "Early-Warning Alert", sub: "Rapid Intensification (+30kt / 24h) detection" },
    forecast: { title: "Forecast", sub: "Multi-horizon intensity prediction from sequence satellite frames" },
    about: { title: "About & Data", sub: "Dataset composition, validation setup, and model architecture" },
  };

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const activeContent = document.getElementById(`tab-${target}`);
      if (activeContent) activeContent.classList.add("active");

      const meta = TAB_META[target];
      if (meta && topbarTitle && topbarSub) {
        topbarTitle.textContent = meta.title;
        topbarSub.textContent = meta.sub;
      }

      // Lazy load tab data
      if (target === "history") loadHistory();
      if (target === "ri") loadRIAlert();
      if (target === "forecast") loadForecast();
      if (target === "about") loadAbout();
    });
  });

  // Initial Load
  initSamples();

  async function initSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      const select = document.getElementById("sample-select");
      select.innerHTML = "";

      data.samples.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.img_name;
        opt.textContent = `${s.img_name} (${s.cat_name}, ${s.kmph} km/h)`;
        select.appendChild(opt);
      });

      if (data.samples.length > 0) {
        currentSampleName = data.samples[0].img_name;
        select.value = currentSampleName;
        loadAssessment(currentSampleName);
      }

      select.addEventListener("change", (e) => {
        currentSampleName = e.target.value;
        loadAssessment(currentSampleName);
      });
    } catch (err) {
      console.error("Error initializing samples:", err);
    }
  }

  // --- TAB 1: ASSESS A STORM ---
  async function loadAssessment(imgName) {
    try {
      const res = await fetch(`/api/assess?img_name=${encodeURIComponent(imgName)}`);
      const d = await res.json();

      document.getElementById("img-ir").src = d.ir_base64;
      document.getElementById("img-raw").src = d.raw_base64;
      document.getElementById("img-overlay").src = d.overlay_base64;

      // Alert Banner
      const banner = document.getElementById("assess-alert-banner");
      let icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>', headline = "LOW IMMEDIATE THREAT", levelClass = "good";
      if (d.pred_idx >= 5) {
        icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'; headline = "HIGH-IMPACT — Review Evacuation Readiness"; levelClass = "critical";
      } else if (d.pred_idx >= 3) {
        icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'; headline = "SIGNIFICANT — Monitor Closely"; levelClass = "warning";
      }
      const lowConf = d.confidence < 0.65;
      if (levelClass === "good" && lowConf) {
        icon = ICON_ALERT_TRIANGLE;
        headline = "LIKELY LOW THREAT — Confidence Too Low To Confirm";
        levelClass = "warning";
      }
      const confNote = lowConf ? " — verify manually" : "";
      banner.className = `alert-banner ${levelClass}`;
      banner.innerHTML = `<span style="display:inline-flex;flex-shrink:0;">${icon}</span> <strong>${headline}</strong> &bull; ${(d.confidence * 100).toFixed(0)}% confidence${confNote}`;

      // Metrics Grid
      const metricsContainer = document.getElementById("assess-metrics");
      let metricsHTML = `
        <div class="metric-card">
          <div class="metric-label">Predicted Category</div>
          <div class="metric-value">${d.pred_cat_name}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Predicted Wind Speed</div>
          <div class="metric-value">${d.pred_kmph} km/h</div>
          <div class="metric-delta">&plusmn;${d.wind_std_kmph} km/h</div>
        </div>
      `;
      if (d.pred_pressure) {
        metricsHTML += `
          <div class="metric-card">
            <div class="metric-label">Predicted Pressure</div>
            <div class="metric-value">${d.pred_pressure} mb</div>
            <div class="metric-delta">&plusmn;${d.pressure_std_mb} mb</div>
          </div>
        `;
      }
      if (d.true_cat_name) {
        metricsHTML += `
          <div class="metric-card">
            <div class="metric-label">Actual Category</div>
            <div class="metric-value">${d.true_cat_name}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Actual Wind Speed</div>
            <div class="metric-value">${d.true_kmph} km/h</div>
          </div>
        `;
      }
      metricsContainer.innerHTML = metricsHTML;

      // Category Probabilities Chart
      renderProbabilitiesChart(d.cat_names, d.mean_probs, d.std_probs, d.pred_idx);
    } catch (err) {
      console.error("Error loading assessment:", err);
    }
  }

  function renderProbabilitiesChart(labels, means, stds, predIdx) {
    const ctx = document.getElementById("chart-probabilities").getContext("2d");
    const barColors = labels.map((_, i) => i === predIdx ? CAT_ORANGE : (SEQ_RAMP[i] || CAT_BLUE));

    if (chartProb) chartProb.destroy();

    chartProb = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Probability (MC-Dropout mean)",
          data: means,
          backgroundColor: barColors,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 24 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(1)}%`
            }
          },
          highlightLabel: {
            enabled: true,
            index: predIdx,
            color: CAT_ORANGE,
            formatter: (v) => `${(v * 100).toFixed(0)}%`
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            grid: { color: "#f1f5f9" },
            ticks: { color: "#64748b" }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#334155", font: { weight: "600" } }
          }
        }
      }
    });
  }

  // --- TAB 2: HISTORICAL PRECEDENT ---
  async function loadHistory() {
    if (!currentSampleName) return;
    try {
      const res = await fetch(`/api/history?img_name=${encodeURIComponent(currentSampleName)}`);
      const d = await res.json();
      const grid = document.getElementById("history-grid");
      grid.innerHTML = "";

      d.matches.forEach(m => {
        const card = document.createElement("div");
        card.className = "history-match-card";
        const title = m.storm
          ? `${m.storm}${m.display_date ? " \u00b7 " + m.display_date : ""}`
          : "Unidentified sample (no storm ID)";
        card.innerHTML = `
          <img src="${m.ir_base64}" alt="${title}">
          <div class="match-info">
            <div class="match-name">${title}</div>
            <div class="match-badge">${m.percentile}th percentile match</div>
            <div class="match-details">${m.cat_name}, ${m.kmph} km/h</div>
          </div>
        `;
        grid.appendChild(card);
      });
    } catch (err) {
      console.error("Error loading history:", err);
    }
  }

  // --- TAB 3: EARLY-WARNING ALERT ---
  async function loadRIAlert() {
    try {
      const res = await fetch("/api/ri_alert");
      const d = await res.json();

      const metricsContainer = document.getElementById("ri-metrics");
      if (d.episodes && d.episodes.length > 0) {
        const ep = d.episodes[0];
        metricsContainer.innerHTML = `
          <div class="metric-card status-critical">
            <div class="metric-label">RI Status</div>
            <div class="metric-value">TRIGGERED</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Intensification</div>
            <div class="metric-value">+${ep.delta} kt</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">${ep.duration_hours} h</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Peak Intensity</div>
            <div class="metric-value">${d.peak_wind_kt} kt &bull; SuCS</div>
          </div>
        `;
      } else {
        metricsContainer.innerHTML = `
          <div class="metric-card status-good">
            <div class="metric-label">RI Status</div>
            <div class="metric-value">NOT TRIGGERED</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Peak Intensity</div>
            <div class="metric-value">${d.peak_wind_kt != null ? d.peak_wind_kt + " kt" : "\u2014"}</div>
          </div>
        `;
      }

      // Amphan Timeline Chart
      const ctx = document.getElementById("chart-ri").getContext("2d");
      if (chartRI) chartRI.destroy();

      chartRI = new Chart(ctx, {
        type: "line",
        data: {
          labels: d.times,
          datasets: [{
            label: "Sustained Wind Speed (kt)",
            data: d.winds,
            borderColor: CAT_BLUE,
            backgroundColor: "rgba(2, 132, 199, 0.08)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.2,
            pointRadius: 3,
            pointBackgroundColor: CAT_BLUE,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Wind Speed: ${ctx.raw} kt`
              }
            }
          },
          scales: {
            y: {
              grid: { color: "#f1f5f9" },
              ticks: { color: "#64748b" },
              title: { display: true, text: "Wind Speed (kt)", color: "#64748b" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#64748b", maxTicksLimit: 10 }
            }
          }
        }
      });
    } catch (err) {
      console.error("Error loading RI alert:", err);
    }
  }

  // --- TAB 4: FORECAST ---
  async function loadForecast(pickedStorm = null, anchorIdx = null) {
    try {
      let url = "/api/forecast";
      const params = [];
      if (pickedStorm) params.push(`storm=${encodeURIComponent(pickedStorm)}`);
      if (anchorIdx !== null) params.push(`anchor_idx=${anchorIdx}`);
      if (params.length > 0) url += "?" + params.join("&");

      const res = await fetch(url);
      const d = await res.json();

      // Storm Selectbox
      const select = document.getElementById("storm-select");
      if (select.children.length === 0) {
        select.innerHTML = "";
        d.storms.forEach(st => {
          const opt = document.createElement("option");
          opt.value = st;
          opt.textContent = d.val_storms.includes(st) ? `${st} (held out \u2014 validation)` : `${st} (used in training)`;
          select.appendChild(opt);
        });
        select.value = d.picked_storm;

        select.addEventListener("change", (e) => {
          loadForecast(e.target.value, 0);
        });
      } else {
        select.value = d.picked_storm;
      }

      // Slider
      const slider = document.getElementById("anchor-slider");
      slider.max = d.total_anchors - 1;
      slider.value = d.anchor_idx;
      document.getElementById("anchor-caption").textContent = `Forecasting from ${d.anchor_dt} (Anchor ${d.anchor_idx + 1} of ${d.total_anchors})`;

      slider.oninput = (e) => {
        loadForecast(d.picked_storm, e.target.value);
      };

      // Callout
      const callout = document.getElementById("forecast-heldout-callout");
      if (d.is_held_out) {
        callout.className = "callout info";
        callout.innerHTML = `<span class="callout-icon"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg></span> <div><strong>${d.picked_storm}</strong> was held out entirely during training — forecast below reflects true generalization.</div>`;
      } else {
        callout.className = "callout warning";
        callout.innerHTML = `<span class="callout-icon"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg></span> <div><strong>${d.picked_storm}</strong> was used during training — select Phet or Nilofar for held-out validation.</div>`;
      }

      // Thumbnails
      const thumbsGrid = document.getElementById("forecast-thumbs");
      thumbsGrid.innerHTML = "";
      d.thumbnails.forEach(t => {
        const item = document.createElement("div");
        item.className = "thumb-item";
        item.innerHTML = `
          <img src="${t.ir_base64}" alt="${t.dt}">
          <div class="thumb-caption">${t.dt}</div>
        `;
        thumbsGrid.appendChild(item);
      });

      // Metrics
      const metricsContainer = document.getElementById("forecast-metrics");
      let metricsHTML = `
        <div class="metric-card">
          <div class="metric-label">Current Anchor Wind</div>
          <div class="metric-value">${d.anchor_kmph} km/h</div>
          <div class="metric-delta">${d.anchor_cat}</div>
        </div>
      `;
      ["6", "12", "24"].forEach(h => {
        const item = d.horizons[h];
        if (item) {
          const deltaSign = item.delta_kmph >= 0 ? "+" : "";
          const actualText = item.actual_kmph ? ` &bull; actual ${item.actual_kmph} km/h` : "";
          metricsHTML += `
            <div class="metric-card">
              <div class="metric-label">+${h}h Horizon</div>
              <div class="metric-value">${item.pred_kmph} km/h</div>
              <div class="metric-delta">${deltaSign}${item.delta_kmph} km/h vs now (${item.pred_cat}${actualText})</div>
            </div>
          `;
        }
      });
      metricsContainer.innerHTML = metricsHTML;

      // Chart: Observed vs Forecast
      const ctx = document.getElementById("chart-forecast").getContext("2d");
      if (chartForecast) chartForecast.destroy();

      chartForecast = new Chart(ctx, {
        type: "line",
        data: {
          labels: d.full_times,
          datasets: [
            {
              label: "Observed Wind Speed (km/h)",
              data: d.full_winds,
              borderColor: CAT_BLUE,
              borderWidth: 2.5,
              pointRadius: 2,
              tension: 0.2,
            },
            {
              label: "Forecast Trajectory",
              data: d.full_times.map((t, idx) => {
                const fcIdx = d.fc_times.indexOf(t);
                return fcIdx !== -1 ? d.fc_winds[fcIdx] : null;
              }),
              borderColor: CAT_ORANGE,
              borderDash: [6, 4],
              borderWidth: 2.5,
              pointRadius: 5,
              pointBackgroundColor: CAT_ORANGE,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: true, position: "top" }
          },
          scales: {
            y: {
              grid: { color: "#f1f5f9" },
              ticks: { color: "#64748b" },
              title: { display: true, text: "Wind Speed (km/h)", color: "#64748b" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#64748b", maxTicksLimit: 12 }
            }
          }
        }
      });

      // Live RI Banner
      const riBanner = document.getElementById("forecast-ri-banner");
      if (d.rt_episodes && d.rt_episodes.length > 0) {
        const ep = d.rt_episodes[0];
        riBanner.innerHTML = `
          <div class="alert-banner critical">
            <span style="display:inline-flex;flex-shrink:0;"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg></span>
            <div><strong>RAPID INTENSIFICATION ALERT:</strong> Forecast predicts +${ep.delta_kt}kt over ${ep.duration_hours}h crossing threshold (+30kt / 24h).</div>
          </div>
        `;
      } else {
        riBanner.innerHTML = `
          <div class="callout info">
            <span class="callout-icon"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg></span>
            <div>No Rapid Intensification threshold crossing predicted in this forecast window (+30kt / 24h).</div>
          </div>
        `;
      }

    } catch (err) {
      console.error("Error loading forecast:", err);
    }
  }

  // --- TAB 5: ABOUT & DATA ---
  async function loadAbout() {
    try {
      const res = await fetch("/api/about");
      const d = await res.json();

      const metricsContainer = document.getElementById("about-metrics");
      metricsContainer.innerHTML = `
        <div class="metric-card">
          <div class="metric-label">Labeled Images</div>
          <div class="metric-value">${d.total_samples}</div>
          <div class="metric-delta">${d.n_kaggle} Kaggle, ${d.n_hursat} HURSAT, ${d.n_mosdac} MOSDAC</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">IMD Categories</div>
          <div class="metric-value">${d.composition.length}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Validation Setup</div>
          <div class="metric-value">2 Held-Out Storms</div>
          <div class="metric-delta">${d.val_accuracy}% val acc (epoch ${d.val_epoch})</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Architecture</div>
          <div class="metric-value">CycloneNet</div>
          <div class="metric-delta">2-channel IR/Raw backbone</div>
        </div>
      `;

      // Composition Chart
      const ctxComp = document.getElementById("chart-composition").getContext("2d");
      if (chartComposition) chartComposition.destroy();

      chartComposition = new Chart(ctxComp, {
        type: "bar",
        data: {
          labels: d.composition.map(c => c.name),
          datasets: [{
            label: "Labeled Images",
            data: d.composition.map(c => c.count),
            backgroundColor: d.composition.map((c, i) => SEQ_RAMP[c.cat_idx] || CAT_BLUE),
            borderRadius: 6,
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: "#f1f5f9" }, ticks: { color: "#64748b" } },
            y: { grid: { display: false }, ticks: { color: "#334155", font: { weight: "600" } } }
          }
        }
      });

      // Training History Chart
      if (d.training_history && d.training_history.length > 0) {
        const ctxTrain = document.getElementById("chart-training").getContext("2d");
        if (chartTraining) chartTraining.destroy();

        chartTraining = new Chart(ctxTrain, {
          type: "line",
          data: {
            labels: d.training_history.map(h => h.epoch),
            datasets: [
              {
                label: "Train Accuracy",
                data: d.training_history.map(h => h.train_acc),
                borderColor: CAT_BLUE,
                borderWidth: 2.5,
              },
              {
                label: "Validation Accuracy (Held-out storms)",
                data: d.training_history.map(h => h.val_acc),
                borderColor: CAT_ORANGE,
                borderWidth: 2.5,
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, position: "top" } },
            scales: {
              y: { beginAtZero: true, max: 1.0, grid: { color: "#f1f5f9" }, ticks: { color: "#64748b" } },
              x: { grid: { display: false }, ticks: { color: "#64748b" } }
            }
          }
        });
      }
    } catch (err) {
      console.error("Error loading about data:", err);
    }
  }
});
