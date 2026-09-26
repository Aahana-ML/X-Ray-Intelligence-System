const imageInput = document.getElementById("imageInput");
const analyzeButton = document.getElementById("analyzeButton");
const preview = document.getElementById("preview");
const results = document.getElementById("results");

const classSelect = document.getElementById("classSelect");
const explainButton = document.getElementById("explainButton");
const gradcamResult = document.getElementById("gradcamResult");

let selectedFile = null;


// ===============================
// IMAGE PREVIEW
// ===============================

imageInput.addEventListener("change", () => {

    selectedFile = imageInput.files[0];

    if (!selectedFile) {
        return;
    }

    const imageURL = URL.createObjectURL(selectedFile);

    preview.innerHTML = `
        <img
            src="${imageURL}"
            class="preview-image"
        >
    `;
});


// ===============================
// PREDICTION
// ===============================

analyzeButton.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an X-ray image first.");
        return;
    }

    selectedFile = file;

    const formData = new FormData();

    formData.append("file", file);

    results.innerHTML = "<p>Analyzing X-ray...</p>";

    try {

        const response = await fetch(
            "http://localhost:9000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        // Check whether FastAPI returned an error
        if (!response.ok) {
            throw new Error(
                `Prediction failed: ${response.status}`
            );
        }

        const data = await response.json();

        displayResults(data.predictions);

        populateClassSelector(data.predictions);

    } catch (error) {

        results.innerHTML =`
        <p>Prediction error:</p>
        <p>${error.message}</p>
    `;
            

        console.error("Prediction error:", error);
    }
});


// ===============================
// DISPLAY PREDICTION RESULTS
// ===============================

function displayResults(predictions) {

    results.innerHTML = "<h2>Analysis Results</h2>";

    for (const [label, result] of Object.entries(predictions)) {

        const percentage =
            (result.score * 100).toFixed(1);

        const status =
            result.detected
                ? "Detected"
                : "Not detected";

        const statusClass =
            result.detected
                ? "detected"
                : "not-detected";

        results.innerHTML += `
            <div class="result">

                <strong>${label}</strong>

                <span class="${statusClass}">
                    ${percentage}% — ${status}
                </span>

            </div>
        `;
    }
}


// ===============================
// POPULATE GRAD-CAM DROPDOWN
// ===============================

function populateClassSelector(predictions) {

    classSelect.innerHTML =
        `<option value="">Select abnormality</option>`;

    for (const label of Object.keys(predictions)) {

        const option = document.createElement("option");

        option.value = label;
        option.textContent = label;

        classSelect.appendChild(option);
    }
}


// ===============================
// GENERATE GRAD-CAM
// ===============================

explainButton.addEventListener("click", async () => {

    if (!selectedFile) {

        alert("Please analyze an X-ray first.");

        return;
    }

    const className = classSelect.value;

    if (!className) {

        alert("Please select an abnormality first.");

        return;
    }

    const formData = new FormData();

    formData.append("file", selectedFile);

    formData.append("class_name", className);

    gradcamResult.innerHTML =
        "<p>Generating Grad-CAM...</p>";

    try {

        const response = await fetch(
            "http://localhost:9000/explain",
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {

            throw new Error(
                `Grad-CAM failed: ${response.status}`
            );
        }

        const blob = await response.blob();

        const imageURL =
            URL.createObjectURL(blob);

        gradcamResult.innerHTML = `
            <h3>${className} — Model Attention</h3>

            <img
                src="${imageURL}"
                class="gradcam-image"
            >
        `;

    } catch (error) {

        gradcamResult.innerHTML =
            "<p>Could not generate Grad-CAM.</p>";

        console.error("Grad-CAM error:", error);
    }
});