// script.js
async function predict() {
  const text = document.getElementById("text-input").value;

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const result = await response.json();
    console.log(result, 'hi'  )

    document.getElementById("result").innerText = 
      `Prediction: ${result.prediction}\nConfidence: ${result.confidence}`;
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("result").innerText = "Prediction failed.";
  }
}


