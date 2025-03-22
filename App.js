import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css"; // Import CSS for styling

const App = () => {
    const [imageSrc, setImageSrc] = useState(null);
    const [feedback, setFeedback] = useState("Waiting for ID card...");
    const [showWebcam, setShowWebcam] = useState(true);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (showWebcam) {
            setLoading(true);
            setTimeout(() => setLoading(false), 2000); // Simulate camera startup delay
            const interval = setInterval(() => {
                axios.get("http://localhost:5000/get_quality_feedback")
                    .then(response => {
                        if (!imageSrc) setFeedback(response.data.message);
                    })
                    .catch(() => setFeedback("Error fetching feedback"));
            }, 500);
            return () => clearInterval(interval);
        }
    }, [showWebcam, imageSrc]);

    const fetchCapturedCard = () => {
        axios.get("http://localhost:5000/get_captured_card")
            .then(response => {
                if (response.data.id_card) {
                    setImageSrc(`data:image/jpeg;base64,${response.data.id_card}`);
                    setShowWebcam(false);
                    setFeedback(""); // Hide feedback after capturing
                }
            })
            .catch(() => alert("Error fetching image"));
    };

    useEffect(() => {
        const interval = setInterval(fetchCapturedCard, 500);
        return () => clearInterval(interval);
    }, []);

    const handleRetake = () => {
        setLoading(true);
        axios.post("http://localhost:5000/retake")
            .then(() => {
                setImageSrc(null);
                setShowWebcam(true);
                setFeedback("Waiting for ID card...");
                setTimeout(() => setLoading(false), 2000);
            })
            .catch(() => alert("Error triggering retake"));
    };

    return (
        <div className="container">
            <h2>ID Card Detection</h2>
            {loading ? (
                <div className="loader-container">
                    <div className="loader"></div>
                    <p>Loading Camera...</p>
                </div>
            ) : showWebcam ? (
                <img src="http://localhost:5000/video_feed" alt="Video Feed" className="video-feed" />
            ) : (
                <div className="captured-container">
                    <img src={imageSrc} alt="Captured ID Card" className="captured-image" />
                    <button className="retake-button" onClick={handleRetake}>Retake</button>
                </div>
            )}
            {showWebcam && <p className="feedback">{feedback}</p>}
        </div>
    );
};

export default App;
