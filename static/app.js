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

  // IMD Category Palette
  const SEQ_RAMP = ["#bae6fd", "#7dd3fc", "#38bdf8", "#0284c7", "#0369a1", "#1d4ed8", "#1e40af", "#1e3a8a"];
  const CAT_ORANGE = "#f97316";
  const CAT_BLUE = "#0284c7";

  // Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const activeContent = document.getElementById(`tab-${target}`);
      if (activeContent) activeContent.classList.add("active");

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
      let icon = "✅", headline = "LOW IMMEDIATE THREAT", levelClass = "good";
      if (d.pred_idx >= 5) {
        icon = "🚨"; headline = "HIGH-IMPACT — Review Evacuation Readiness"; levelClass = "critical";
      } else if (d.pred_idx >= 3) {
        icon = "⚠️"; headline = "SIGNIFICANT — Monitor Closely"; levelClass = "warning";
      }
      const confNote = d.confidence < 0.65 ? " (low model confidence — verify manually)" : "";
      banner.className = `alert-banner ${levelClass}`;
      banner.innerHTML = `<span style="font-size:1.25rem;">${icon}</span> <strong>${headline}</strong> &bull; ${(d.confidence * 100).toFixed(0)}% confidence${confNote}`;

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
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(1)}%`
            }
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
        card.innerHTML = `
          <img src="${m.ir_base64}" alt="${m.img_name}">
          <div class="match-info">
            <div class="match-name">${m.img_name}</div>
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
          <div class="metric-card">
            <div class="metric-label">RI Status</div>
            <div class="metric-value" style="color:var(--status-critical-text);">TRIGGERED</div>
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
          opt.textContent = d.val_storms.includes(st) ? `${st} (held out — validation)` : st;
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
        callout.innerHTML = `<span class="callout-icon">ℹ️</span> <div><strong>${d.picked_storm}</strong> was held out entirely during training — forecast below reflects true generalization.</div>`;
      } else {
        callout.className = "callout warning";
        callout.innerHTML = `<span class="callout-icon">⚠️</span> <div><strong>${d.picked_storm}</strong> was used during training — select Phet or Nilofar for held-out validation.</div>`;
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
            <span style="font-size:1.25rem;">🚨</span>
            <div><strong>RAPID INTENSIFICATION ALERT:</strong> Forecast predicts +${ep.delta_kt}kt over ${ep.duration_hours}h crossing threshold (+30kt / 24h).</div>
          </div>
        `;
      } else {
        riBanner.innerHTML = `
          <div class="callout info">
            <span class="callout-icon">ℹ️</span>
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
