async function askAgent() {
    let videoId = document.getElementById("videoId").value.trim();
    let question = document.getElementById("question").value.trim();

    document.getElementById("answerBox").textContent = "Loading...";
    document.getElementById("summaryBox").textContent = "";

    const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, question: question })
    });

    const data = await res.json();

    if (data.ok) {
        document.getElementById("answerBox").textContent = data.answer || "No answer";
        document.getElementById("summaryBox").textContent = data.summary || "No summary";
    } else {
        document.getElementById("answerBox").textContent = "Error from backend";
    }
}


