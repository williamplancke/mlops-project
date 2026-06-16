const apiBase = "/api";
const form = document.querySelector("#predictionForm");
const statusEl = document.querySelector("#apiStatus");
const formError = document.querySelector("#formError");
const advancedJson = document.querySelector("#advancedJson");
const loadDefaults = document.querySelector("#loadDefaults");
const resultElements = {
  netProfit: document.querySelector("#netProfit"),
  profit: document.querySelector("#profit"),
  damageProbability: document.querySelector("#damageProbability"),
  expectedDamage: document.querySelector("#expectedDamage"),
  rawResponse: document.querySelector("#rawResponse"),
  incidentBadge: document.querySelector("#incidentBadge"),
};

const formatCurrency = (value) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(value);

const setStatus = (message, ok = true) => {
  statusEl.textContent = message;
  statusEl.classList.toggle("offline", !ok);
};

const errorMessage = (error) =>
  error instanceof Error ? error.message : "Unexpected error.";

const fetchJson = async (url, options) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
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
  resultElements.netProfit.textContent = formatCurrency(data.expected_net_profit);
  resultElements.profit.textContent = formatCurrency(data.predicted_profit);
  resultElements.damageProbability.textContent = `${Math.round(data.damage_probability * 100)}%`;
  resultElements.expectedDamage.textContent = formatCurrency(data.expected_damage_amount);
  resultElements.rawResponse.textContent = JSON.stringify(data, null, 2);

  resultElements.incidentBadge.className = data.predicted_damage_incident
    ? "incident risk"
    : "incident clear";
  resultElements.incidentBadge.textContent = data.predicted_damage_incident
    ? "Damage incident likely"
    : "Damage incident unlikely";
};

const loadFeatureDefaults = async () => {
  const data = await fetchJson(`${apiBase}/features`);
  advancedJson.value = JSON.stringify(data.defaults, null, 2);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";

  try {
    const data = await fetchJson(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: collectFeatures() }),
    });
    renderPrediction(data);
  } catch (error) {
    formError.textContent = errorMessage(error);
  }
});

loadDefaults.addEventListener("click", async () => {
  formError.textContent = "";
  try {
    await loadFeatureDefaults();
  } catch (error) {
    formError.textContent = errorMessage(error);
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
