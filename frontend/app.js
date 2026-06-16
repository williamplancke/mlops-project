const apiBase = "/api";
const form = document.querySelector("#predictionForm");
const statusEl = document.querySelector("#apiStatus");
const formError = document.querySelector("#formError");
const advancedJson = document.querySelector("#advancedJson");
const loadDefaults = document.querySelector("#loadDefaults");

const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(value);

const setStatus = (message, ok = true) => {
  statusEl.textContent = message;
  statusEl.style.background = ok ? "#edf6f2" : "#fff0ec";
  statusEl.style.color = ok ? "#155b45" : "#b2452d";
};

const collectFeatures = () => {
  const features = {};
  new FormData(form).forEach((value, key) => {
    if (value !== "") {
      features[key] = Number(value);
    }
  });

  const advanced = advancedJson.value.trim();
  if (advanced) {
    Object.assign(features, JSON.parse(advanced));
  }

  return features;
};

const renderPrediction = (data) => {
  document.querySelector("#netProfit").textContent = formatCurrency(data.expected_net_profit);
  document.querySelector("#profit").textContent = formatCurrency(data.predicted_profit);
  document.querySelector("#damageProbability").textContent = `${Math.round(data.damage_probability * 100)}%`;
  document.querySelector("#expectedDamage").textContent = formatCurrency(data.expected_damage_amount);
  document.querySelector("#rawResponse").textContent = JSON.stringify(data, null, 2);

  const badge = document.querySelector("#incidentBadge");
  badge.className = data.predicted_damage_incident ? "incident risk" : "incident clear";
  badge.textContent = data.predicted_damage_incident
    ? "Damage incident likely"
    : "Damage incident unlikely";
};

const loadFeatureDefaults = async () => {
  const response = await fetch(`${apiBase}/features`);
  if (!response.ok) {
    throw new Error("Could not load feature defaults.");
  }
  const data = await response.json();
  advancedJson.value = JSON.stringify(data.defaults, null, 2);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: collectFeatures() }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderPrediction(data);
  } catch (error) {
    formError.textContent = error.message;
  }
});

loadDefaults.addEventListener("click", async () => {
  formError.textContent = "";
  try {
    await loadFeatureDefaults();
  } catch (error) {
    formError.textContent = error.message;
  }
});

fetch(`${apiBase}/health`)
  .then((response) => {
    if (!response.ok) {
      throw new Error();
    }
    setStatus("API online", true);
  })
  .catch(() => setStatus("API offline", false));
