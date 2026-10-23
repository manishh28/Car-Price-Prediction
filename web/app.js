(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const scriptUrl = document.currentScript?.src || window.location.href;
  const form = $("form");
  const brand = $("brand");
  const model = $("model");
  const year = $("year");
  const km = $("km");
  const button = form.querySelector(".go");
  const error = $("error");
  const owners = {
    "1": "First Owner",
    "2": "Second Owner",
    "3": "Third Owner",
    "4": "Fourth & Above Owner"
  };
  let db;

  const money = (value) => {
    const lakh = value / 100000;
    if (lakh >= 100) return `₹${(lakh / 100).toFixed(1)} crore`;
    if (lakh >= 1) return `₹${(Math.round(lakh * 10) / 10).toFixed(1)} lakh`;
    return `₹${Math.round(value / 1000) * 1000}`;
  };

  function selected(name) {
    return form.querySelector(`input[name="${name}"]:checked`)?.value;
  }

  function choose(name, value) {
    const option = Array.from(form.querySelectorAll(`input[name="${name}"]`))
      .find((input) => input.value === value);
    if (option) option.checked = true;
  }

  function findBrand(value) {
    if (!db) return null;
    const normalized = value.trim().toLocaleLowerCase();
    return db.brands.find((item) => item.toLocaleLowerCase() === normalized) || null;
  }

  function clearModelDetails() {
    ["mileage", "cc", "bhp", "seats"].forEach((name) => {
      form.elements[name].value = "";
    });
    choose("owner", "1");
    choose("fuel", "Petrol");
    choose("gear", "Manual");
  }

  function syncModelList() {
    const matchedBrand = findBrand(brand.value);
    model.value = "";
    clearModelDetails();
    if (!matchedBrand) {
      model.disabled = true;
      model.placeholder = "Pick a brand first";
      $("model-list").replaceChildren();
      return;
    }

    brand.value = matchedBrand;
    const names = db.models_by_brand[matchedBrand] || [];
    model.disabled = false;
    model.placeholder = names.length ? `Search ${names.length} ${matchedBrand} models` : "No models listed";
    $("model-list").replaceChildren(...names.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      return option;
    }));
  }

  function loadModelDetails() {
    if (!db) return;
    const entry = db.models[model.value];
    if (!entry || entry.b !== findBrand(brand.value)) {
      clearModelDetails();
      return;
    }

    const spec = Object.fromEntries(db.spec_keys.map((key, i) => [key, entry.s[i]]));
    form.elements.mileage.value = spec.mileage_value ?? "";
    form.elements.cc.value = spec.engine_value ?? "";
    form.elements.bhp.value = spec.max_power_value ?? "";
    form.elements.seats.value = spec.seats ?? "";
    choose("owner", Object.keys(owners).find((key) => owners[key] === spec.owner) || "1");
    choose("fuel", spec.fuel || "Petrol");
    choose("gear", spec.transmission || "Manual");
  }

  function showError(message, field) {
    error.textContent = message;
    error.hidden = false;
    if (field) {
      field.classList.add("invalid");
      field.focus();
    }
  }

  async function loadData() {
    const response = await fetch(new URL("data.json", scriptUrl), { cache: "no-cache" });
    if (!response.ok) throw new Error(`Vehicle data request failed (${response.status}).`);
    db = await response.json();
    if (!Array.isArray(db.brands) || !db.models_by_brand || !db.models || !Array.isArray(db.spec_keys)) {
      throw new Error("Vehicle data file is incomplete.");
    }

    $("brand-list").replaceChildren(...db.brands.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      return option;
    }));
    year.innerHTML = '<option value="">Choose year</option>' +
      Array.from({ length: new Date().getFullYear() - db.year_min + 1 }, (_, i) => new Date().getFullYear() - i)
        .map((value) => `<option value="${value}">${value}</option>`).join("");

    $("foot-stats").textContent =
      `Prices are estimates, not offers. Based on ${db.n_listings.toLocaleString("en-IN")} used-car listings. ` +
      "Check the car in person before you pay.";
  }

  brand.addEventListener("input", syncModelList);
  brand.addEventListener("change", syncModelList);
  model.addEventListener("change", loadModelDetails);
  model.addEventListener("input", () => {
    if (db?.models[model.value]) loadModelDetails();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.hidden = true;
    form.querySelectorAll(".invalid").forEach((field) => field.classList.remove("invalid"));

    if (!db) return showError("The vehicle list is unavailable right now. Please try again shortly.");
    const matchedBrand = findBrand(brand.value);
    if (!matchedBrand) return showError("Choose a brand from the suggestions.", brand);
    const entry = db.models[model.value];
    if (!entry || entry.b !== matchedBrand) return showError(`Choose a model listed under ${matchedBrand}.`, model);
    if (!year.value) return showError("Choose the registration year.", year);
    if (km.value === "" || !Number.isFinite(Number(km.value)) || Number(km.value) < 0 || Number(km.value) > 1500000) {
      return showError("Enter the kilometres driven (between 0 and 15,00,000).", km);
    }

    const payload = {
      name: model.value,
      year: Number(year.value),
      km_driven: Number(km.value),
      owner: owners[selected("owner")] || "First Owner",
      fuel: selected("fuel") || "Petrol",
      transmission: selected("gear") || "Manual",
      mileage_value: form.elements.mileage.value ? Number(form.elements.mileage.value) : null,
      engine_value: form.elements.cc.value ? Number(form.elements.cc.value) : null,
      max_power_value: form.elements.bhp.value ? Number(form.elements.bhp.value) : null,
      seats: form.elements.seats.value ? Number(form.elements.seats.value) : null
    };

    button.disabled = true;
    button.textContent = "Estimating…";
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The estimate could not be calculated.");

      $("r-car").textContent = `${payload.year} ${matchedBrand} ${model.value}, ${payload.km_driven.toLocaleString("en-IN")} km`;
      $("r-price").textContent = money(result.predicted_price);
      $("r-range").textContent = "A guide based on your car details and used-car listings.";
      $("r-fine").textContent =
        "The final price can vary with condition, service history, accident history, location, and local demand.";
      $("empty").hidden = true;
      $("result").hidden = false;
      if (window.matchMedia("(max-width: 900px)").matches) {
        $("result").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (exception) {
      showError(exception.message);
    } finally {
      button.disabled = false;
      button.textContent = "Get the price";
    }
  });

  loadData().catch((exception) => {
    console.error("Vehicle data could not be initialized:", exception);
    showError("We couldn't load the vehicle list. Please try again in a moment.");
  });
})();
