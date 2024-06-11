async function getPrediction(features) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features }),
    });
    const data = await response.json();
    return data.prediction;
}

// Example usage
const features = [/* Your feature values here */];
getPrediction(features).then(prediction => {
    console.log('Prediction:', prediction);
});
